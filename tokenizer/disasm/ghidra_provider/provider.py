"""Ghidra-based disassembly provider using pyghidra (headless, no IPC).

pyghidra gives direct access to the Ghidra Java API from CPython via JPype.
This provider translates Ghidra's Instruction/Register/Scalar objects into
Capstone-compatible adapter objects so existing ArchitectureProviders work
unchanged.

Requirements:
    pip install pyghidra
    GHIDRA_INSTALL_DIR env var or pass install_dir to pyghidra.start()
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional

from tokenizer.disasm import DisassemblyProvider, MetadataLookup
from tokenizer.disasm.ghidra_provider.decode_helper import _GhidraDecodeHelper
from tokenizer.disasm.ghidra_provider.metadata_lookup import GhidraMetadataLookup
from tokenizer.disasm.ghidra_provider.mnemonic import (
    _GHIDRA_MNEMONIC_ALIASES,
    _RegisterMap,
    _split_ghidra_mnemonic,
)
from tokenizer.disasm.ghidra_provider.prefix_build import (
    _compute_fp_type,
    _ghidra_processor_to_architecture,
)
from tokenizer.disasm.ghidra_provider.switch_table_walker import (
    walk_switch_tables_for_function,
)
from tokenizer.disasm.ghidra_views.unnamed_rename import (
    compute_binary_identity_hash,
    placeholder_renamed_name,
)
from tokenizer.disasm.types import FpType


# ---------------------------------------------------------------------------
# JVM startup
# ---------------------------------------------------------------------------

# Shenandoah uncommits committed heap back to the OS during idle periods,
# which eases cgroup-memory pressure between tasks in long-lived workers
# (Ghidra holds the JVM process-globally and only releases per-task state
# in `close()`). Tuned for fast uncommit response while the worker waits
# on its next task.
_GHIDRA_JVM_VMARGS = (
    "-XX:+UnlockExperimentalVMOptions",
    "-XX:+UseShenandoahGC",
    "-XX:ShenandoahUncommitDelay=1000",
    "-XX:ShenandoahGuaranteedGCInterval=10000",
)


def _ensure_jvm_started() -> None:
    """Idempotently boot the Ghidra JVM with our vmargs injected.

    pyghidra exposes vmargs only via the ``HeadlessPyGhidraLauncher``
    object; the convenience ``pyghidra.start()`` builds its own launcher
    internally and gives no hook to add vmargs. Routing every provider
    through this helper keeps the vmargs configuration in one place and
    preserves the original ``pyghidra.started()`` short-circuit.
    """
    import pyghidra

    if pyghidra.started():
        return
    launcher = pyghidra.HeadlessPyGhidraLauncher()
    launcher.add_vmargs(*_GHIDRA_JVM_VMARGS)
    launcher.start()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GhidraDisassemblyProvider(DisassemblyProvider):
    """Disassembly provider backed by Ghidra via pyghidra (headless, no IPC).

    Usage::

        provider = GhidraDisassemblyProvider(Path("binary"))
        provider.build_cfg()   # runs Ghidra auto-analysis
        for addr, name, func in provider.iter_functions():
            ...
    """

    def __init__(
        self,
        binary_path: Path,
        duplicate_function_dump_path: Path | None = None,
    ) -> None:
        import jpype.config
        import pyghidra

        # Prevent JVM from hanging on Python exit (known JPype issue)
        jpype.config.destroy_jvm = False

        _ensure_jvm_started()

        self.binary_path = binary_path
        # ``None`` -> dump disabled (zero work in iter_functions). When
        # set, the path is the absolute destination for the per-binary
        # duplicate-function-metadata pickle; the orchestrator module
        # owns the file write + parent-mkdir.
        self._duplicate_function_dump_path: Path | None = duplicate_function_dump_path
        # 16-byte per-binary identity hash. Computed ONCE here (folding
        # the binary path + dataset sidecar JSON content when present);
        # threaded into every ``_GhidraFunctionView`` so the placeholder
        # rename helper pays a single XOR + b64encode on each
        # DEFAULT-source function. See
        # ``tokenizer.disasm.ghidra_views.unnamed_rename`` for the
        # rationale + scheme.
        self._binary_id_hash: bytes = compute_binary_identity_hash(binary_path)

        # pyghidra defaults `project_location` to the binary's parent
        # directory, which is the read-only `--source-already-staged`
        # bind-mount in the SLURM container. Redirect into the
        # writable ephemeral root (`/app/out-tmp` if the container
        # provides it, else `/tmp`). Per-binary subdir keyed off the
        # binary's parent dir name (unique per variant in our corpus
        # layout) so concurrent variants don't collide on the
        # `<binary-name>_ghidra` project name pyghidra derives.
        out_tmp_root = Path("/app/out-tmp")
        project_root = out_tmp_root if out_tmp_root.is_dir() else Path("/tmp")
        project_location = project_root / "ghidra-projects" / binary_path.parent.name
        project_location.mkdir(parents=True, exist_ok=True)
        self._project_location = project_location

        # open_program is the simplest API: import, auto-analyze, return
        # a FlatProgramAPI context manager.  We defer analysis (analyze=False)
        # so build_cfg() controls when it happens.
        self._ctx = pyghidra.open_program(
            binary_path,
            project_location=project_location,
            analyze=False,
        )
        self._flat_api = self._ctx.__enter__()
        self._program = self._flat_api.getCurrentProgram()
        self._fm = self._program.getFunctionManager()
        self._listing = self._program.getListing()
        self._memory = self._program.getMemory()
        self._reg_map = _RegisterMap(self._program)
        self._analyzed = False
        self._closed = False
        # Owned-view scaffolding lazily initialized by iter_functions on
        # first call (so close() can null out program references without
        # leaving stale views behind).
        self._function_view: Optional[Any] = None
        # `iter_functions` populates this map with `entry_addr -> Ghidra
        # Function` for every function it visits; `iter_switch_tables`
        # consumes it to look up the underlying Java handle from the
        # FunctionView (no more `function._raw` unwrap dance).
        self._funcs_by_entry: dict[int, Any] = {}

    def close(self) -> None:
        """Unload the current Program + close the Project so the
        next task in this worker (worker processes are reused
        unless `always_restart_worker=True`) can `open_program`
        again without tripping ``GhidraScriptUtil initialized
        multiple times`` or accumulating analysis threads. Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        # `pyghidra.open_program` returns a generator-based context
        # manager. Driving its `__exit__` is the documented way to
        # release the imported Program, the in-memory project, and
        # the analysis-thread pool. Errors on close (e.g. if a
        # prior failure left Ghidra in a half-state) must not
        # mask the original task's exception.
        try:
            self._ctx.__exit__(None, None, None)
        except Exception:
            pass
        # Drop Java-object references so the next provider in this
        # worker can acquire fresh ones without lingering CFG/listing
        # objects pinning the previous program.
        self._flat_api = None
        self._program = None
        self._fm = None
        self._listing = None
        self._memory = None
        self._reg_map = None
        self._ctx = None
        # Drop owned-view scaffolding too so the next task starts
        # from a clean slate.
        self._function_view = None
        self._funcs_by_entry = {}

    def build_cfg(self) -> None:
        """Run Ghidra's auto-analysis (the equivalent of CFGFast)."""
        self._flat_api.analyzeAll(self._program)
        self._analyzed = True

    def get_text_section_bounds(self) -> tuple[int, int]:
        for block in self._memory.getBlocks():
            name = str(block.getName())
            if name == ".text":
                start = int(block.getStart().getOffset())
                return start, start + int(block.getSize())
        return 0, 0

    def parse_data_sections(
        self,
        sections: list[str] | None = None,
        output_csv_path: str | None = None,
    ) -> dict[str, list[str]]:
        if sections is None:
            sections = [".rodata"]

        all_entries: list[dict[str, str]] = []
        addr_dict: dict[str, list[str]] = {}

        for block in self._memory.getBlocks():
            name = str(block.getName())
            if name not in sections:
                continue
            if name == ".rodata" and block.isInitialized() and not block.isWrite():
                size = int(block.getSize())
                if size <= 0:
                    continue
                start_addr = block.getStart()
                buf = bytearray(size)
                block.getBytes(start_addr, buf)
                data = bytes(buf)
                base_addr = int(start_addr.getOffset())

                for match in re.finditer(b"[\x20-\x7e]{4,}\x00", data):
                    s = match.group().rstrip(b"\x00").decode("utf-8", errors="ignore")
                    start = base_addr + match.start()
                    entry = {
                        "section": ".rodata",
                        "start": hex(start),
                        "end": hex(start + len(s) + 1),
                        "value": f'"{s}"',
                    }
                    all_entries.append(entry)
                    addr_dict[entry["start"]] = [entry["end"], entry["section"], entry["value"]]

        if output_csv_path:
            csv_path = Path(output_csv_path)
            consts_path = csv_path.parent / f"{csv_path.stem.replace('_output', '')}_consts.txt"
        else:
            consts_path = Path("parsed_constants.txt")

        with open(consts_path, "w") as f:
            for e in all_entries:
                f.write(f"{e['start']} - {e['end']}: {e['section']}: {e['value']}\n")

        print(f"Parsed {len(all_entries)} .rodata constants with exact addresses into {consts_path}")
        return addr_dict

    def create_metadata_lookup(self) -> MetadataLookup:
        # Thread the precomputed per-binary identity hash so the lookup
        # applies the same ``SourceType.DEFAULT`` placeholder rename to
        # ``meta.name`` that ``iter_functions`` already applies to the
        # yielded ``name`` slot. Without this the JSON metadata column
        # (``local_funcs[].name`` etc.) keeps raw Ghidra ``FUN_<hex>`` /
        # ``LAB_<hex>`` placeholders, which the function-names sidecar
        # then inherits via the per-CSV callee extractor.
        return GhidraMetadataLookup(self._program, self._fm, self._binary_id_hash)

    def function_count(self) -> int:
        return int(self._fm.getFunctionCount())

    def iter_functions(self) -> Iterable[tuple[int, str, "_GhidraFunctionView"]]:
        """Iterate functions, yielding (addr, name, function_view) triples.

        Internal model:
            * One reusable ``_GhidraFunctionView`` is mutated each step to
              point at the current Ghidra Function (lazy/reuse contract,
              per ``tokenizer/disasm/types.py``). The reused view is the
              source of truth for consumers.
            * ``self._funcs_by_entry`` is populated as we iterate so
              ``iter_switch_tables(function)`` can look up the Ghidra
              Java handle via ``function.entry`` (no more
              ``isinstance/_raw`` dance).
        """
        assert self._analyzed, "Analysis not run -- call build_cfg() first"

        from ghidra.program.model.block import SimpleBlockModel
        from ghidra.util.task import TaskMonitor

        # Lazily initialize the owned-view scaffolding: decode helper +
        # function view (reused across every iter step). The scaffolding
        # is rebuilt per ``iter_functions`` call so each call gets a
        # fresh cursor (defensive: callers that re-enter mid-iteration
        # would otherwise share cursor state).
        decode = _GhidraDecodeHelper(self._program, self._reg_map)
        block_model = SimpleBlockModel(self._program)
        monitor = TaskMonitor.DUMMY
        from tokenizer.disasm.ghidra_views import _GhidraFunctionView

        function_view = _GhidraFunctionView(
            arch=decode.arch,
            program=self._program,
            listing=self._listing,
            reg_map=self._reg_map,
            decode=decode,
            block_model=block_model,
            monitor=monitor,
            binary_id_hash=self._binary_id_hash,
        )
        self._function_view = function_view

        # Collect + sort by name (legacy contract). The DEFAULT-source
        # placeholder rename is applied here so the same name flows
        # through every downstream channel — the sort key, the
        # ``duplicate_function_dump`` collision detector, the yielded
        # tuple's ``name`` slot, and ``function_view.name``. See
        # ``tokenizer.disasm.ghidra_views.unnamed_rename`` for the
        # rationale + scheme.
        funcs: list[tuple[int, str, Any]] = []
        for func in self._fm.getFunctions(True):
            raw_name = str(func.getName())
            source = func.getSymbol().getSource()
            name = placeholder_renamed_name(raw_name, source, self._binary_id_hash)
            addr = int(func.getEntryPoint().getOffset())
            funcs.append((addr, name, func))

        # Optional debug dump: when the provider was constructed with a
        # ``duplicate_function_dump_path``, hand the collected funcs
        # list to the orchestrator before sorting so it can detect
        # name-collisions and snapshot each colliding function's
        # 5-layer-deep Ghidra metadata. The hook is gated on the path
        # being non-None - zero work when off.
        if self._duplicate_function_dump_path is not None:
            from tokenizer.disasm.ghidra_provider.duplicate_function_dump import (
                write_duplicate_function_dump,
            )

            write_duplicate_function_dump(
                funcs,
                binary_name=self.binary_path.name,
                output_path=self._duplicate_function_dump_path,
            )

        # Reset the entry-point map for this iter call (callers that
        # re-iterate get a fresh map, no stale entries).
        self._funcs_by_entry = {}

        for addr, name, ghidra_func in sorted(funcs, key=lambda t: t[1]):
            body = ghidra_func.getBody()
            block_iter = block_model.getCodeBlocksContaining(body, monitor)

            # Count code blocks so the reused FunctionView can expose an
            # O(1) ``len(blocks)`` once we advance it. Functions with zero
            # code blocks are skipped.
            block_count = 0
            while block_iter.hasNext():
                block_iter.next()
                block_count += 1

            if block_count == 0:
                continue

            function_view._advance(ghidra_func, block_count)
            self._funcs_by_entry[addr] = ghidra_func
            # ``name`` is the post-``placeholder_renamed_name`` value
            # collected above, so the yielded tuple's ``name`` slot
            # matches ``function_view.name`` — both downstream channels
            # (``main_loop`` -> ``canonical_function_name`` reads the
            # tuple slot; the view reads its cursor) see the same
            # rename.
            yield addr, name, function_view

    # ----------------------------------------------------------------------
    # Per-operand FP-immediate detection
    # ----------------------------------------------------------------------
    def operand_fp_type(self, ghidra_insn: Any, operand_index: int) -> Optional[FpType]:
        """Return the typed ``FpType`` for ``operand_index`` of ``ghidra_insn``.

        Returns one of ``FpType.FLOAT16`` / ``FpType.BFLOAT16`` /
        ``FpType.FLOAT32`` / ``FpType.FLOAT64`` / ``FpType.FLOAT80`` /
        ``FpType.FLOAT128`` when the operand is FP-typed (per Ghidra's
        ``OperandType.FLOAT`` bitmask + the per-ISA BFloat16 mnemonic
        table at width=2) and ``None`` otherwise. Width derivation
        order:

        1. Inspect each ``getOpObjects(i)`` element. For ``Register``
           operands, ``Register.getBitLength() / 8``. For ``Scalar``
           operands, ``Scalar.bitLength() / 8``. Take the largest value
           seen (x87 ``fld dword ptr [...]`` carries an FP-tagged
           memory operand whose size is the load size).
        2. Map width-in-bytes through ``_FP_WIDTH_TO_TYPE``.
        3. At width=2, reclassify Float16 -> BFloat16 when the
           instruction's mnemonic appears in the per-ISA table
           (``_bfloat16_mnemonic_for_arch``). SLEIGH does not currently
           tag bfloat16 distinctly, so the mnemonic-based reclassification
           is the only signal available.
        4. Widths outside ``_FP_WIDTH_TO_TYPE`` return ``None`` (the
           classifier then routes through step 11 of the precedence list
           rather than emitting a malformed ``floatXX``).

        Implementation note: the owned-view decode path stamps this type
        on every operand view's ``fp_type`` at instruction-translation
        time, so operand tokenizers don't need to call this method
        per-operand. The public method stays on the provider for direct
        callers and for symmetry with the Phase 1.B.1
        provider-interface contract.
        """
        arch = _ghidra_processor_to_architecture(self._program)
        try:
            raw_mnemonic = str(ghidra_insn.getMnemonicString())
        except Exception:
            raw_mnemonic = ""
        base_mnemonic, _, _ = _split_ghidra_mnemonic(raw_mnemonic)
        base_mnemonic = _GHIDRA_MNEMONIC_ALIASES.get(base_mnemonic, base_mnemonic)
        return _compute_fp_type(ghidra_insn, operand_index, arch, base_mnemonic)

    # ----------------------------------------------------------------------
    # Switch-table recovery
    # ----------------------------------------------------------------------
    def iter_switch_tables(self, function: Any) -> Iterable[tuple[int, list[int]]]:
        """Yield ``(jump_table_addr, [target_block_addrs])`` for ``function``.

        Walks every instruction in the function's body; for any computed-jump
        instruction, follows the ``COMPUTED_JUMP`` references to gather the
        list of resolved targets, and locates the backing pointer-array data
        in rodata (the table's address) via the instruction's outbound READ
        references when present. Tables are yielded in dispatch-instruction
        order; duplicate-table elision is the consumer's responsibility.

        Phase-1 scope: pragmatic recovery from Ghidra's reference graph.
        A processor-specific ``SwitchAnalyzer`` cross-check is a Phase 2
        refinement (Phase 2.C.1 jump-table-analysis pass).

        ``function`` accepts any object with an ``entry`` attribute - the
        typed ``FunctionView`` yielded by ``iter_functions`` or any
        future shape that exposes the function's entry-point address.
        The Ghidra Java handle is looked up via ``self._funcs_by_entry``
        (populated lazily by ``iter_functions`` as it walks each
        function); calling ``iter_switch_tables`` without first
        iterating ``iter_functions`` on the same provider returns an
        empty iterator.

        The actual body walk lives in
        ``tokenizer.disasm.ghidra_provider.switch_table_walker``; this
        method's concern is resolving the ``function.entry`` ->
        Ghidra ``Function`` handle and delegating.
        """
        if function is None:
            return
        try:
            entry = int(function.entry)
        except AttributeError:
            return
        ghidra_function = self._funcs_by_entry.get(entry)
        if ghidra_function is None:
            return
        yield from walk_switch_tables_for_function(
            ghidra_function, self._listing
        )
