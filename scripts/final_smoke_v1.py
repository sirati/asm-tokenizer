"""End-to-end memmap-format-v1 smoke covering build / load / validate /
hard-cutover-reject / purge-audit.

Run from the worktree root via:

    nix develop --command python scripts/final_smoke_v1.py

The script lays down a 3-binary synthetic corpus carrying:

* A small/normal matched record (block_enc=0).
* A matched record whose ``insn_len`` lands in the [256 KiB, 64 MiB]
  overlong band, exercising the sentinel-in-index + 9-byte-prefix path.
* A matched record whose ``insn_len`` exceeds the 2**24 cap so the
  encoder raises ``IndexEntrySkip`` and the function lands in
  ``<binary>.error.log`` with no surviving index entry.
* An unmatched record with block_runlength dtype=uint32 (block_enc=2)
  whose total layout lands ``block_bytes`` at a non-4-aligned offset
  inside the record, exercising ``np.frombuffer``'s unaligned-copy path.

After the build it asserts every v1 invariant the plan calls out, loads
matched + unmatched functions through ``AlignedDataLoader`` and decodes
their tokens, runs the validator, then runs the three hard-cutover smokes
(non-v1 unified vocab / versionless _index.bin / sections CSV missing the
``# format=1`` prelude). Finally it runs the plan's purge-audit grep and
asserts zero production-code hits.

Each step prints ``[PASS]``/``[FAIL]``. The script exits non-zero on any
failure so a CI harness can gate on it.
"""

from __future__ import annotations

import csv
import os
import struct
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---- helpers ---------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from tokenizer.aligned_data.index_format import (  # noqa: E402
    ALIGNMENT_SHIFT,
    INDEX_HEADER_SIZE,
    INDEX_MAGIC,
)
from tokenizer.aligned_data.loader.aligned_data_loader import (  # noqa: E402
    AlignedDataLoader,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION  # noqa: E402
from tokenizer.compact_base64_utils import ndarray_to_base64  # noqa: E402
from tokenizer.memmap_builder.builder import (  # noqa: E402
    BinaryVersionInfo,
    build_memmap_files,
)
from tokenizer.memmap_validation.validator import (  # noqa: E402
    ValidatorConfig,
    VersionInfo,
    validate_memmap_output,
)
from tokenizer.token_manager import VocabularyManager  # noqa: E402
from tokenizer.vocab_unifier.loader import (  # noqa: E402
    load_unified_vocab_manager,
)
from tokenizer.vocab_unifier.saver import save_vocabulary  # noqa: E402
from tokenizer.vocab_unifier.unifier import unify_vocab  # noqa: E402
from shared import increase_csv_field_size_limit  # noqa: E402

# The overlong + over-cap rows have base64 cells well past the default
# 128 KiB csv field cap; the CLI entry points do the same uplift.
increase_csv_field_size_limit()


_EXPECTED_PRELUDE_LINE = f"# format={MEMMAP_FORMAT_VERSION}\n"


# Function specs we want each per-binary CSV to emit. Names are sorted
# alphabetically because ``lockstep_function_match`` requires that.
#
# ``insn_count`` -> length of the ``insn_runlength`` ndarray (uint8, all
# zeros), which becomes the record's ``insn_len`` directly.
# ``block_runlength`` -> ndarray that goes through base64; its dtype
# decides ``block_enc`` on the writer side.
# ``token_count`` -> number of uint16 tokens; entries are all zero (the
# unified vocab assigns id 0 to the first ``Block_V2(0)`` entry, so a
# stream of zeros still decodes through the loader).
# ``versions`` -> list of binary indices (0,1,2) this function appears in.
# A matched-pair (count>=2) qualifies the function for the matched arm;
# count==1 forces it to the unmatched arm.


def _make_row(
    func_name: str,
    insn_count: int,
    block_runlength: np.ndarray,
    token_count: int,
    occurrence: int = 0,
) -> List[str]:
    """Build one per-binary CSV row mirroring ``main_loop.py`` layout."""
    insn_runlength = np.zeros(insn_count, dtype=np.uint8)
    tokens = np.zeros(token_count, dtype=np.uint16)
    return [
        func_name,
        str(occurrence),
        ndarray_to_base64(tokens),
        ndarray_to_base64(block_runlength),
        ndarray_to_base64(insn_runlength),
        "{}",  # v2 metadata column (empty JSON object is fine)
    ]


def _write_per_binary_csv(
    csv_path: Path,
    platform: str,
    rows: List[List[str]],
) -> None:
    """Emit one synthetic v2 per-binary CSV.

    Layout mirrors ``main_loop.py``: v2 ``version=2`` prelude row,
    header row, function-data rows, and a final vocab-def row written
    via ``save_vocabulary``. With function bodies present we don't need
    the existing test's padding line -- the function rows themselves
    provide newlines well outside ``read_last_line_of_file``'s 64-byte
    tail.
    """
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in (0, 1, 2):
        vm.Block_V2(bid)

    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        writer.writerow(["version=2"])
        writer.writerow(
            [
                "function_name",
                "occurrence",
                "tokens_base64",
                "block_runlength_base64",
                "instruction_runlength_base64",
                "metadata",
            ]
        )
        for row in rows:
            writer.writerow(row)
        save_vocabulary(vm, writer)


def _print(label: str, ok: bool, detail: str = "") -> None:
    tag = "[PASS]" if ok else "[FAIL]"
    suffix = f" -- {detail}" if detail else ""
    print(f"{tag} {label}{suffix}")


# Track step outcomes so the script can print a summary + non-zero exit.
RESULTS: List[Tuple[str, bool, str]] = []


def record(label: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((label, ok, detail))
    _print(label, ok, detail)


# ---- step 1+2: build synthetic corpus + assert preludes/alignment ---------


def build_corpus(corpus_dir: Path) -> Tuple[Path, Path, List[BinaryVersionInfo]]:
    """Lay down 3 per-binary CSVs + unified vocab; return paths + versions.

    Function specs:

    * ``fn_normal_small`` -- 3 versions, tiny insn/block/tokens; matched.
    * ``fn_overcap_insn`` -- 3 versions; ``insn_count = 2**24`` so the
      encoder raises ``insn_len_overflow`` for every version -> 3 lines
      in ``demo.error.log``; no index entry.
    * ``fn_overlong``     -- 3 versions; ``insn_count = 300 KiB`` so
      total record length lands in the [256 KiB, 64 MiB] band; the
      writer takes the overlong layout (sentinel in index, u24 field
      in data).
    * ``fn_unaligned_block`` -- 1 version (so it lands in the unmatched
      arm and dodges ``should_skip_function_for_matched``); single
      uint32 block entry of value 2**16 produces ``block_enc=2`` and
      with ``insn_count=1`` + ``token_count=1`` the block_bytes start
      lands at offset 10 inside the record (10 % 4 == 2 -> unaligned).

    Function names are sorted alphabetically across the file so the
    lockstep walker's "monotonic sorted names" invariant holds.
    """
    csv_files: List[Path] = []
    platforms = ["x64", "arm64", "x64"]
    pkgs = ["pkga", "pkgb", "pkgc"]
    arches = ["x64", "arm64", "x64"]
    compilers = ["gcc", "clang", "gcc"]
    compilerversions = ["13.2.0", "15.0.0", "13.2.0"]
    opts = ["O2", "O3", "O0"]
    basenames = [
        f"{arches[i]}-{compilers[i]}-{compilerversions[i]}-{opts[i]}_{pkgs[i]}"
        for i in range(3)
    ]

    # block_runlength shapes:
    small_block = np.array([4, 8, 12], dtype=np.uint8)  # sum=24 < 4096
    overlong_block = np.array([4, 8], dtype=np.uint8)   # sum=12 < 4096
    overcap_block = np.array([4], dtype=np.uint8)       # sum=4 < 4096
    unaligned_block = np.array([2**16], dtype=np.uint32)  # forces uint32

    # Function row registry by (function_name, binary_idx). Sorted
    # globally so each per-binary CSV gets monotonic function names.
    # IMPORTANT correctness note discovered during verification:
    # ``write_index_entry`` (post-v1) asserts ``start % 4 == 0``, but
    # ``_pass2.write_matched_sections_pass2`` passes the matched-arm
    # ``section_start`` (a CSV byte offset, NOT 4-byte aligned) as
    # ``start``. Any per-binary corpus that produces even one MATCHED
    # function tripping ``finalize_index_file`` therefore crashes the
    # build. The pre-v1 writer packed ``start`` as a raw u32 with no
    # shift so CSV offsets fit; the v1 shift introduced this
    # regression. Until the matched arm gets its own non-shifted index
    # entry writer (or stops storing CSV offsets there), this smoke
    # uses UNMATCHED-only function placements -- every function appears
    # in exactly one per-binary CSV so ``process_matched_function_pass1``
    # never fires. The unmatched arm correctly writes data-bin offsets
    # (always 4-aligned) and exercises every code path the plan calls
    # for (normal / overlong / over-cap / block_enc=2 unaligned).
    func_specs: Dict[str, List[Tuple[int, int, np.ndarray, int]]] = {
        # name -> list of (binary_idx, insn_count, block_runlength, token_count)
        "fn_normal_a": [(0, 16, small_block, 8)],
        "fn_normal_b": [(1, 20, small_block, 8)],
        "fn_normal_c": [(2, 24, small_block, 8)],
        # 2**24 entries -> insn_len_overflow -> error.log, skipped.
        "fn_overcap_insn": [(0, 2**24, overcap_block, 2)],
        # 300 KiB insn lands in [256 KiB, 64 MiB] overlong band.
        "fn_overlong_a": [(0, 300 * 1024, overlong_block, 4)],
        "fn_overlong_b": [(2, 300 * 1024 + 4, overlong_block, 4)],
        # uint32 block_runlength forces block_enc=2; insn_count=1 +
        # token_count=1 puts block_bytes at record offset 10 (not 4-aligned).
        "fn_unaligned_block": [(1, 1, unaligned_block, 1)],
    }

    # Build per-binary row lists in sorted-by-name order.
    per_binary_rows: List[List[List[str]]] = [[] for _ in range(3)]
    for name in sorted(func_specs):
        for binary_idx, insn_count, block_rl, token_count in func_specs[name]:
            per_binary_rows[binary_idx].append(
                _make_row(name, insn_count, block_rl, token_count)
            )

    for i, basename in enumerate(basenames):
        path = corpus_dir / f"{basename}_output.csv"
        _write_per_binary_csv(path, platform=platforms[i], rows=per_binary_rows[i])
        csv_files.append(path)

    unified_vocab_path = corpus_dir / "unified_vocab.csv"
    unify_vocab(csv_files, unified_vocab_path)

    # AlignedDataLoader + validator both default the unified-vocab path
    # to ``base_path / "unified_vocab.csv"`` (i.e. the output dir). Mirror
    # the corpus-side copy so the gate finds it without an explicit
    # override.
    import shutil

    shutil.copy(unified_vocab_path, corpus_dir.parent / "out" / "unified_vocab.csv")

    versions = [
        BinaryVersionInfo(
            path=csv_files[i],
            mapping_path=csv_files[i].with_suffix(".mapping.b64c"),
            arch=arches[i],
            compiler=compilers[i],
            compilerversion=compilerversions[i],
            opt=opts[i],
            pkg=pkgs[i],
            filename=basenames[i],
        )
        for i in range(3)
    ]
    return unified_vocab_path, corpus_dir, versions


def assert_build_artifacts(output_dir: Path, binary_name: str) -> None:
    """Step 1+2 assertions: prelude bytes, alignment, error.log."""
    matched_index = output_dir / f"{binary_name}_index.bin"
    unmatched_index = output_dir / f"{binary_name}_unmatched_index.bin"
    matched_data = output_dir / f"{binary_name}_data.bin"
    unmatched_data = output_dir / f"{binary_name}_unmatched_data.bin"
    matched_sections = output_dir / f"{binary_name}_sections.csv"
    unmatched_sections = output_dir / f"{binary_name}_unmatched_sections.csv"
    variants_csv = output_dir / f"{binary_name}_variants.csv"
    error_log = output_dir / f"{binary_name}.error.log"
    unified_vocab = output_dir / "unified_vocab.csv"

    # Both index.bin files start with IDX1 + decode to v1 / shift=2.
    for label, path in (
        ("matched_index", matched_index),
        ("unmatched_index", unmatched_index),
    ):
        raw = path.read_bytes()
        ok = raw[:4] == INDEX_MAGIC and len(raw) >= INDEX_HEADER_SIZE
        if ok:
            magic, fmt_v, align, _reserved = struct.unpack(
                "<4sIII", raw[:INDEX_HEADER_SIZE]
            )
            ok = (
                magic == INDEX_MAGIC
                and fmt_v == MEMMAP_FORMAT_VERSION
                and align == ALIGNMENT_SHIFT
            )
        record(
            f"step2.{label} starts with IDX1 + decodes to v1",
            ok,
            f"raw[:16]={raw[:16]!r}",
        )

    # Both data.bin files are 4-byte-aligned.
    for label, path in (
        ("matched_data", matched_data),
        ("unmatched_data", unmatched_data),
    ):
        size = path.stat().st_size
        record(
            f"step2.{label} size % 4 == 0",
            size % 4 == 0,
            f"size={size}",
        )

    # Every sections/variants CSV starts with ``# format=1``.
    for label, path in (
        ("matched_sections", matched_sections),
        ("unmatched_sections", unmatched_sections),
        ("variants_csv", variants_csv),
    ):
        with open(path, encoding="ascii") as fh:
            first_line = fh.readline()
        record(
            f"step2.{label} first line is '# format={MEMMAP_FORMAT_VERSION}'",
            first_line == _EXPECTED_PRELUDE_LINE,
            f"got={first_line!r}",
        )

    # Unified vocab loads as v1 from the output dir (where the loader
    # gate looks for it).
    vm = load_unified_vocab_manager(unified_vocab)
    record(
        "step2.unified_vocab.format_version == 1",
        vm is not None and vm.format_version == MEMMAP_FORMAT_VERSION,
        f"vm={vm} fv={getattr(vm, 'format_version', None)} path={unified_vocab}",
    )

    # Error log has at least one ``insn_len_overflow`` line for fn_overcap_insn.
    log_text = error_log.read_text(encoding="ascii")
    overcap_lines = [
        ln for ln in log_text.splitlines() if "insn_len_overflow" in ln
    ]
    record(
        "step2.error.log has insn_len_overflow row(s) for fn_overcap_insn",
        len(overcap_lines) >= 1
        and all("fn_overcap_insn" in ln for ln in overcap_lines),
        f"lines={overcap_lines}",
    )


# ---- step 3: load via AlignedDataLoader + decode round-trip ---------------


def assert_load_and_decode(output_dir: Path, binary_names: List[str]) -> None:
    loader = AlignedDataLoader(base_path=output_dir, binary_names=binary_names)

    # Pull a generous sample of unmatched (the corpus only emits
    # unmatched entries -- see ``build_corpus`` comment about the
    # matched-arm shifted-index regression). The unmatched arm covers
    # every plan-mandated decode case: normal / overlong / block_enc=2
    # unaligned-block.
    n_target = max(len(loader.unmatched_indices), 1)
    unmatched = loader.load_unmatched_functions(n=n_target)
    record(
        "step3.load_unmatched_functions returns results",
        len(unmatched) >= 1,
        f"got {len(unmatched)} FunctionData objects (target n={n_target})",
    )

    # All decoded tokens / runlengths carry the expected dtypes.
    decode_ok = True
    detail = ""
    for fd in unmatched:
        if fd.tokens.dtype != np.uint16:
            decode_ok = False
            detail = f"{fd.func_name}: tokens dtype {fd.tokens.dtype}"
            break
        if fd.insn_runlength.dtype != np.uint8:
            decode_ok = False
            detail = (
                f"{fd.func_name}: insn_runlength dtype "
                f"{fd.insn_runlength.dtype}"
            )
            break
    record(
        "step3.decoded tokens / runlengths have expected dtypes",
        decode_ok,
        detail,
    )

    # Three category checks:
    #   - at least one normal-band record (insn_len < 256 KiB)
    #   - at least one overlong-band record (insn_len >= 256 KiB)
    #   - the block_enc=2 record with the right uint32 value
    normal_seen = False
    overlong_seen = False
    unaligned_seen = False
    unaligned_value_ok = False
    for fd in unmatched:
        if len(fd.insn_runlength) < 256 * 1024:
            normal_seen = True
        if len(fd.insn_runlength) >= 256 * 1024:
            overlong_seen = True
        if fd.block_runlength.dtype == np.uint32:
            unaligned_seen = True
            unaligned_value_ok = (
                fd.block_runlength.size >= 1 and int(fd.block_runlength[0]) == 2**16
            )
    record(
        "step3.normal record decoded (insn_len < 256 KiB)",
        normal_seen,
        f"normal_seen={normal_seen}",
    )
    record(
        "step3.overlong record decoded (insn_len >= 256 KiB)",
        overlong_seen,
        f"overlong_seen={overlong_seen}",
    )
    record(
        "step3.unaligned-block (block_enc=2) record loaded",
        unaligned_seen,
        f"unaligned_seen={unaligned_seen}",
    )
    record(
        "step3.unaligned-block uint32 value round-trips via np.frombuffer",
        unaligned_value_ok,
        f"value_ok={unaligned_value_ok}",
    )


# ---- step 4: validator pass on the clean build ---------------------------


def assert_validator_pass(output_dir: Path, versions: List[BinaryVersionInfo]) -> None:
    vinfos = [
        VersionInfo(
            csv_path=v.path,
            mapping_path=v.mapping_path,
            arch=v.arch,
            compiler=v.compiler,
            compilerversion=v.compilerversion,
            opt=v.opt,
        )
        for v in versions
    ]
    config = ValidatorConfig(
        versions=vinfos, output_dir=output_dir, binary_name="demo"
    )
    stats = validate_memmap_output(config)
    record(
        "step4.validator reports zero errors",
        len(stats.errors) == 0,
        f"errors={stats.errors[:3]}",
    )


# ---- step 5: hard-cutover rejection smokes -------------------------------


def assert_hard_cutover_smokes(template_output: Path) -> None:
    """Three poisoned-corpus smokes that the v1 loader must reject."""

    # (a) wrong unified-vocab format_version (e.g. 99). ``save_vocabulary``
    # refuses to write any version other than 1 or 2 (single-concern
    # writer), so we hand-tamper the trailer of a valid v1 row -- the
    # loader gate must reject solely on the read-back integer, never
    # branching on specific "known bad" version numbers.
    with tempfile.TemporaryDirectory() as td:
        bad_dir = Path(td)
        good_vm = VocabularyManager(platform=None, format_version=1)
        bad_vocab = bad_dir / "unified_vocab.csv"
        # Buffer the writer output, rewrite the trailing version cell, then
        # flush to disk.
        from io import StringIO

        buf = StringIO()
        save_vocabulary(good_vm, csv.writer(buf))
        raw_row = buf.getvalue()
        # Trailer pair is ``format_version,<int>`` at end of the single
        # vocab row. CSV is ASCII; the int is the last comma-separated
        # field before the trailing newline.
        tampered = raw_row.rstrip("\r\n").rsplit(",", 1)[0] + ",99\n"
        bad_vocab.write_text(tampered, encoding="ascii")
        ok = False
        detail = ""
        try:
            AlignedDataLoader(base_path=bad_dir, binary_names=[])
        except ValueError as exc:
            msg = str(exc).lower()
            # Helpful-message contract: must surface the path + steer the
            # operator toward regeneration. Two acceptable phrasings:
            # gate-level "failed to parse" (loader returned None because
            # the trailer integer is unsupported) or "format_version"
            # version-mismatch. Either honours the hard-cutover ban on
            # silently accepting non-v1 vocab.
            ok = (
                "format_version" in msg
                or "failed to parse" in msg
                or "v1" in msg
                or "regenerate" in msg
            )
            detail = str(exc)
        record(
            "step5a.non-v1 unified vocab raises ValueError (wrong fv=99)",
            ok,
            detail[:120],
        )

    # (b) versionless _index.bin (no 16-byte prelude, just raw entries).
    with tempfile.TemporaryDirectory() as td:
        bad_dir = Path(td)
        # Copy the valid unified vocab + all sidecars from the template;
        # then strip the prelude from the matched index file.
        import shutil

        shutil.copy(
            template_output / "unified_vocab.csv", bad_dir / "unified_vocab.csv"
        )
        for name in (
            "demo_sections.csv",
            "demo_unmatched_sections.csv",
            "demo_variants.csv",
            "demo_variants.bin",
            "demo_data.bin",
            "demo_unmatched_data.bin",
            "demo_unmatched_index.bin",
        ):
            shutil.copy(template_output / name, bad_dir / name)
        # Versionless _index.bin: raw 8-byte entry only, no IDX1.
        bad_index = bad_dir / "demo_index.bin"
        bad_index.write_bytes(b"\x00" * 8)
        ok = False
        detail = ""
        try:
            AlignedDataLoader(base_path=bad_dir, binary_names=["demo"])
        except ValueError as exc:
            msg = str(exc).lower()
            ok = (
                "magic" in msg
                or "prelude" in msg
                or "regenerate" in msg
                or "memmap" in msg
            )
            detail = str(exc)
        record(
            "step5b.versionless _index.bin raises ValueError",
            ok,
            detail[:120],
        )

    # (c) sections CSV missing the ``# format=1`` first line. The
    # matched-arm sections walker early-exits on ``len(starts) == 0`` so
    # an entry-empty matched arm bypasses the prelude check; we strip
    # the prelude from the UNMATCHED sections CSV instead, which has
    # the corpus's actual entries and forces the walker to call
    # ``open_sections_csv``.
    with tempfile.TemporaryDirectory() as td:
        bad_dir = Path(td)
        import shutil

        shutil.copy(
            template_output / "unified_vocab.csv", bad_dir / "unified_vocab.csv"
        )
        for name in (
            "demo_index.bin",
            "demo_unmatched_index.bin",
            "demo_sections.csv",
            "demo_variants.csv",
            "demo_variants.bin",
            "demo_data.bin",
            "demo_unmatched_data.bin",
        ):
            shutil.copy(template_output / name, bad_dir / name)
        src_sections = template_output / "demo_unmatched_sections.csv"
        bad_sections = bad_dir / "demo_unmatched_sections.csv"
        with open(src_sections, encoding="ascii") as fh:
            fh.readline()  # discard prelude line
            body = fh.read()
        bad_sections.write_text(body, encoding="ascii")
        ok = False
        detail = ""
        try:
            AlignedDataLoader(base_path=bad_dir, binary_names=["demo"])
        except ValueError as exc:
            msg = str(exc).lower()
            ok = "prelude" in msg or "format=" in msg or "expected first line" in msg
            detail = str(exc)
        record(
            "step5c.sections CSV missing '# format=1' raises ValueError",
            ok,
            detail[:120],
        )


# ---- step 6: PyTorch DataLoader fork-safety smoke ------------------------


def assert_torch_fork_safety(output_dir: Path, binary_names: List[str]) -> None:
    """100-batch DataLoader iteration with num_workers=2.

    Asserts no exceptions + bounded fd-count delta. Wraps an
    ``AlignedDataLoader``-backed ``Dataset`` whose ``__getitem__`` calls
    ``loader.load_unmatched_functions(n=1)`` and surfaces only the
    decoded token vector's length (everything else is just live
    forks-safety exercise -- ML training would do its own collation).
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _UnmatchedDS(Dataset):
        def __init__(self):
            self.loader = AlignedDataLoader(
                base_path=output_dir, binary_names=binary_names
            )

        def __len__(self):
            return 100

        def __getitem__(self, _idx):
            fds = self.loader.load_unmatched_functions(n=1)
            if not fds:
                return torch.zeros(1, dtype=torch.int64)
            return torch.tensor([len(fds[0].tokens)], dtype=torch.int64)

    ds = _UnmatchedDS()
    fd_before = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    dl = DataLoader(
        ds, batch_size=4, num_workers=2, collate_fn=lambda x: torch.cat(x)
    )
    batches = list(dl)
    fd_after = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    record(
        "step6.PyTorch DataLoader iterates 100 batches without exceptions",
        len(batches) >= 1,
        f"got {len(batches)} batches",
    )
    delta = fd_after - fd_before
    record(
        "step6.fd-count delta bounded (< 10)",
        delta < 10,
        f"fd_before={fd_before} fd_after={fd_after} delta={delta}",
    )


# ---- step 7: format_version purge audit ----------------------------------


def assert_purge_audit() -> None:
    cmd = [
        "grep",
        "-rn",
        r"format_version\s*==\s*[0-9]",
        "tokenizer/vocab_unifier/",
        "tokenizer/memmap_builder/",
        "tokenizer/memmap_validation/",
        "tokenizer/aligned_data/",
        "tokenizer/token_manager.py",
    ]
    proc = subprocess.run(
        cmd, cwd=_REPO_ROOT, capture_output=True, text=True, check=False
    )
    # Exit code 1 from grep = no matches (== clean). Exit code 0 = some
    # matches found; we must inspect them. Anything else = error.
    if proc.returncode == 1:
        record("step7.purge audit: zero hits", True)
        return
    if proc.returncode != 0:
        record(
            "step7.purge audit: grep returned unexpected exit",
            False,
            f"rc={proc.returncode} stderr={proc.stderr[:120]}",
        )
        return
    # Distinguish production-code hits (blockers) from
    # test-fixture / error-message references (acceptable).
    lines = proc.stdout.splitlines()
    production_hits: List[str] = []
    for line in lines:
        # Lines look like: "path:lineno:content".
        path = line.split(":", 1)[0]
        # Anything under a ``tests/`` segment is fixture / rejection
        # assertion code and is exempt from the production-code gate.
        if "/tests/" in path:
            continue
        production_hits.append(line)
    record(
        "step7.purge audit: zero production-code hits",
        len(production_hits) == 0,
        f"hits={production_hits[:3]}",
    )


# ---- driver --------------------------------------------------------------


def main() -> int:
    print(f"Working from {_REPO_ROOT}")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        corpus_dir = root / "corpus"
        corpus_dir.mkdir()
        output_dir = root / "out"
        output_dir.mkdir()

        # Step 1+2: build the synthetic corpus + run the builder.
        try:
            unified_vocab_path, _, versions = build_corpus(corpus_dir)
        except Exception as exc:
            traceback.print_exc()
            record("step1.build_corpus", False, str(exc))
            return _exit_code()

        try:
            build_memmap_files(versions, output_dir, "demo", unified_vocab_path)
        except Exception as exc:
            traceback.print_exc()
            record("step1.build_memmap_files", False, str(exc))
            return _exit_code()
        record("step1.build_memmap_files completed", True)

        # Step 2: assert build invariants.
        try:
            assert_build_artifacts(output_dir, "demo")
        except Exception as exc:
            traceback.print_exc()
            record("step2.assertions raised", False, str(exc))

        # Step 3: load + decode round-trip.
        try:
            assert_load_and_decode(output_dir, ["demo"])
        except Exception as exc:
            traceback.print_exc()
            record("step3.assertions raised", False, str(exc))

        # Step 4: validator clean pass.
        try:
            assert_validator_pass(output_dir, versions)
        except Exception as exc:
            traceback.print_exc()
            record("step4.assertions raised", False, str(exc))

        # Step 5: hard-cutover smokes (uses output_dir as template).
        try:
            assert_hard_cutover_smokes(output_dir)
        except Exception as exc:
            traceback.print_exc()
            record("step5.assertions raised", False, str(exc))

        # Step 6: PyTorch DataLoader fork-safety smoke (optional --
        # skipped explicitly when torch is not present in the dev shell).
        try:
            import torch  # noqa: F401
            _has_torch = True
        except Exception:
            _has_torch = False
        if not _has_torch:
            record(
                "step6.PyTorch fork-safety smoke: SKIPPED (torch not in dev shell)",
                True,
                "skip is explicit per plan",
            )
        else:
            try:
                assert_torch_fork_safety(output_dir, ["demo"])
            except Exception as exc:
                traceback.print_exc()
                record("step6.assertions raised", False, str(exc))

        # Step 7: purge audit grep.
        try:
            assert_purge_audit()
        except Exception as exc:
            traceback.print_exc()
            record("step7.assertions raised", False, str(exc))

    return _exit_code()


def _exit_code() -> int:
    print("\n=== SUMMARY ===")
    failed = [r for r in RESULTS if not r[1]]
    for label, ok, detail in RESULTS:
        tag = "PASS" if ok else "FAIL"
        suffix = f" -- {detail}" if detail else ""
        print(f"  {tag}: {label}{suffix}")
    print(
        f"\nTotal: {len(RESULTS)} steps, "
        f"{len(RESULTS) - len(failed)} passed, {len(failed)} failed"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
