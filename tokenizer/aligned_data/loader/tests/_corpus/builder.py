"""Corpus builder: drive production pass-2 writers + function-names
registry against caller-supplied specs (:mod:`.specs`) to emit the
seven on-disk artefacts a ``BinaryDataset`` reads.

No duplicated layout knowledge; bytes are identical to
``build_memmap_files``. Variant sidecar (``_variants.csv``/``.bin``) is
omitted -- ``load_variants_offset_to_filename`` tolerates a missing
file. Tests needing a real variant record hand-lay the bin and use
:func:`build_corpus_with_registry` to wire variant_refs to its offset.

Direct writer calls (not ``build_memmap_files``) so the fixture stays
free of the full VocabularyManager + per-binary mapping setup that the
end-to-end builder requires -- out of scope for loader integration
tests.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from tokenizer.aligned_data._writers import write_function_binary_data
from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.index_format import write_index_prelude
from tokenizer.aligned_data.matched_sections_bin import SectionWriter
from tokenizer.memmap_builder._pass2 import (
    write_matched_sections_pass2,
    write_unmatched_sections_pass2,
)
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry

from .specs import MatchedFunctionSpec, UnmatchedFunctionSpec


class _StubVariantRegistry:
    """Bare ``.ref(vkey) -> str`` / ``.byte_offset`` surface for the
    pass-2 writers.

    Each unique ``vkey`` gets a deterministic 4-byte-aligned hex
    placeholder (string for the CSV cell, the same integer for the
    BIN's ``variant_ref_offset`` u32). Real ``VariantRegistry``
    couples a unified vocab + a bin write; neither is required to
    exercise the section CSV / BIN wire formats this fixture targets.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._refs: dict = {}

    def _ensure(self, vkey) -> int:
        if vkey not in self._refs:
            self._refs[vkey] = self._counter * 0x10
            self._counter += 1
        return self._refs[vkey]

    def ref(self, vkey) -> str:
        return f"{self._ensure(vkey):x}"

    def byte_offset(self, vkey) -> int:
        return self._ensure(vkey)


@dataclass(frozen=True)
class CorpusPaths:
    """All on-disk artefacts the fixture wrote, plus the originating
    specs so tests can cross-check decoded outputs against expected
    inputs without re-deriving them.
    """

    base_path: Path
    binary_name: str
    matched_specs: Tuple[MatchedFunctionSpec, ...]
    unmatched_specs: Tuple[UnmatchedFunctionSpec, ...]
    function_names_sidecar: Path
    matched_sections_csv: Path
    matched_index_bin: Path
    matched_data_bin: Path
    unmatched_sections_csv: Path
    unmatched_index_bin: Path
    unmatched_data_bin: Path
    sections_bin: Path
    extern_providers_sidecar: Path

    @property
    def matched_function_names(self) -> Tuple[str, ...]:
        return tuple(spec.func_name for spec in self.matched_specs)

    @property
    def unmatched_function_names(self) -> Tuple[str, ...]:
        return tuple(spec.func_name for spec in self.unmatched_specs)

    def read_matched_bin_starts(self) -> np.ndarray:
        """Read the matched_index.bin BIN-section starts back from disk.

        Returns an empty array when the index file is absent or empty
        (matches the loader's behaviour).
        """
        pair = read_csv_section_index_arrays(self.matched_index_bin)
        if pair is None:
            return np.zeros(0, dtype=np.int64)
        starts, _lengths = pair
        return starts


def build_corpus(
    tmp_path: Path,
    binary_name: str,
    *,
    matched: Sequence[MatchedFunctionSpec] = (),
    unmatched: Sequence[UnmatchedFunctionSpec] = (),
) -> CorpusPaths:
    """Lay down a synthetic per-binary memmap output tree on ``tmp_path``.

    Each variant is fed through ``write_function_binary_data`` to
    populate ``_data.bin``; the resulting offsets are handed to the
    pass-2 writers to emit the section CSVs and indices. The
    ``FunctionNamesRegistry`` is finalised between pass 1 and pass 2
    (same ordering as ``builder.py``) so the sidecar carries every
    referenced name. Uses an internal stub variants registry; for
    custom variant_ref wiring use :func:`build_corpus_with_registry`.
    """
    return build_corpus_with_registry(
        tmp_path,
        binary_name,
        matched=matched,
        unmatched=unmatched,
        variants=_StubVariantRegistry(),
    )


def build_corpus_with_registry(
    tmp_path: Path,
    binary_name: str,
    *,
    matched: Sequence[MatchedFunctionSpec] = (),
    unmatched: Sequence[UnmatchedFunctionSpec] = (),
    variants,
) -> CorpusPaths:
    """Same as :func:`build_corpus` but with a caller-supplied variants
    registry (bare ``.ref(vkey) -> str`` surface).

    Used by tests that hand-lay ``_variants.bin`` and need every
    variant_ref cell to point at known on-disk offsets.
    """
    matched_tuple = tuple(matched)
    unmatched_tuple = tuple(unmatched)

    base_path = Path(tmp_path)
    paths = _build_path_bundle(base_path, binary_name)

    registry = FunctionNamesRegistry()
    matched_data_entries, matched_lookup = _emit_matched_data(
        matched_tuple, paths.matched_data_bin, registry
    )
    unmatched_data_entries, unmatched_lookup = _emit_unmatched_data(
        unmatched_tuple, paths.unmatched_data_bin, registry
    )
    function_lookup = {**matched_lookup, **unmatched_lookup}
    matched_func_names = {e["func_name"] for e in matched_data_entries}
    sectioned_func_names = matched_func_names | {
        e["func_name"] for e in unmatched_data_entries
    }
    registry.finalize()
    function_names_sidecar = registry.write_sidecar(base_path, binary_name)

    warn_log = io.StringIO()  # discarded; warn paths not under test here
    extern_providers = ExternProviderRegistry()
    section_writer = SectionWriter(paths.sections_bin)
    try:
        with open(paths.matched_sections_csv, "w", newline="", encoding="ascii") as sf, \
             open(paths.matched_index_bin, "wb") as idxf:
            write_csv_prelude(sf)
            write_matched_sections_pass2(
                matched_data_entries, function_lookup, sf, idxf, warn_log,
                variants, registry,
                section_writer, extern_providers,
                matched_func_names, sectioned_func_names,
                # Spec-driven corpus: no name is duplicated, so every call
                # edge resolves normally (no MISSING-sentinel stamping).
                duplicated_names=set(),
            )
        with open(paths.unmatched_sections_csv, "w", newline="", encoding="ascii") as sf, \
             open(paths.unmatched_index_bin, "wb") as idxf:
            write_csv_prelude(sf)
            write_index_prelude(idxf)  # unmatched keeps the v1 16-byte prelude
            write_unmatched_sections_pass2(
                unmatched_data_entries, function_lookup, sf, idxf, warn_log,
                variants, registry,
                section_writer, extern_providers,
                matched_func_names, sectioned_func_names,
                duplicated_names=set(),
            )
        section_writer.finalize()
    except BaseException:
        section_writer.close()
        raise
    extern_providers_sidecar = extern_providers.write_sidecar(
        base_path, binary_name
    )

    return CorpusPaths(
        base_path=base_path,
        binary_name=binary_name,
        matched_specs=matched_tuple,
        unmatched_specs=unmatched_tuple,
        function_names_sidecar=function_names_sidecar,
        matched_sections_csv=paths.matched_sections_csv,
        matched_index_bin=paths.matched_index_bin,
        matched_data_bin=paths.matched_data_bin,
        unmatched_sections_csv=paths.unmatched_sections_csv,
        unmatched_index_bin=paths.unmatched_index_bin,
        unmatched_data_bin=paths.unmatched_data_bin,
        sections_bin=paths.sections_bin,
        extern_providers_sidecar=extern_providers_sidecar,
    )


@dataclass(frozen=True)
class _PathBundle:
    matched_sections_csv: Path
    matched_index_bin: Path
    matched_data_bin: Path
    unmatched_sections_csv: Path
    unmatched_index_bin: Path
    unmatched_data_bin: Path
    sections_bin: Path


def _build_path_bundle(base_path: Path, binary_name: str) -> _PathBundle:
    return _PathBundle(
        matched_sections_csv=base_path / f"{binary_name}_sections.csv",
        matched_index_bin=base_path / f"{binary_name}_index.bin",
        matched_data_bin=base_path / f"{binary_name}_data.bin",
        unmatched_sections_csv=base_path / f"{binary_name}_unmatched_sections.csv",
        unmatched_index_bin=base_path / f"{binary_name}_unmatched_index.bin",
        unmatched_data_bin=base_path / f"{binary_name}_unmatched_data.bin",
        sections_bin=base_path / f"{binary_name}_sections.bin",
    )


def _project_called_typed(called: Sequence[str]) -> list:
    """Project a name-only callee tuple from the spec into the typed
    ``(name, CallTargetType.LOCAL)`` form pass-2 now consumes.

    Specs don't carry call-type info — fixtures default every callee
    to LOCAL since the loader-corpus tests don't exercise the BIN's
    EXTERN/PLT branches. Phase-4 fixture refactors can widen
    :mod:`.specs` to carry the typed form natively; this projection
    keeps the existing spec API working without forcing a fixture
    rewrite for Phase 3.
    """
    return [(name, CallTargetType.LOCAL) for name in called]


def _emit_matched_data(
    matched: Sequence[MatchedFunctionSpec],
    data_path: Path,
    registry: FunctionNamesRegistry,
) -> Tuple[List[dict], dict]:
    """Write matched-arm ``_data.bin`` records and return the pass-2
    inputs (``matched_data_entries``, ``function_lookup``).

    Stays a pure rearrangement of the production pipeline's pass-1
    outputs -- no parallel encoding logic. The fixture is dedup-free
    (every variant gets a fresh write), so the per-record ``entry_idx``
    is a simple monotonic counter mirroring the file's encounter order;
    the trailer stamps that final count.
    """
    matched_data_entries: List[dict] = []
    lookup: dict = {}
    n_entries = 0
    with open(data_path, "wb") as data_file:
        # Match the production builder's file-level prelude so the
        # loader's prelude-magic assertion fires successfully on test
        # fixtures too.
        from tokenizer.aligned_data.memmap_format import (
            encode_data_bin_prelude,
            encode_data_bin_trailer,
        )
        data_file.write(encode_data_bin_prelude())
        for spec in matched:
            typed_called = _project_called_typed(spec.called)
            version_data = []
            for variant in spec.variants:
                offset, length = write_function_binary_data(
                    data_file, variant.tokens, variant.block_rl, variant.insn_rl,
                    entry_idx=n_entries,
                )
                n_entries += 1
                version_data.append(
                    {
                        "vkey": variant.vkey,
                        "called": set(typed_called),
                        "data_offset": offset,
                        "data_len": length,
                        "token_len": len(variant.tokens),
                    }
                )
                lookup[(spec.func_name, variant.vkey)] = (offset, length, 1)
            matched_data_entries.append(
                {
                    "func_name": spec.func_name,
                    "unique_called": list(typed_called),
                    "extern_libraries": {},
                    "version_data": version_data,
                }
            )
            registry.add(spec.func_name)
            for callee in spec.called:
                registry.add(callee)
        data_file.write(
            encode_data_bin_trailer(n_entries, cursor=data_file.tell())
        )
    return matched_data_entries, lookup


def _emit_unmatched_data(
    unmatched: Sequence[UnmatchedFunctionSpec],
    data_path: Path,
    registry: FunctionNamesRegistry,
) -> Tuple[List[dict], dict]:
    """Write unmatched-arm ``_data.bin`` records via the production writer.

    Same dedup-free encounter-order ``entry_idx`` policy as
    :func:`_emit_matched_data`; the trailer stamps the final count.
    """
    unmatched_data_entries: List[dict] = []
    lookup: dict = {}
    n_entries = 0
    with open(data_path, "wb") as data_file:
        # Match the production builder's file-level prelude so the
        # loader's prelude-magic assertion fires successfully on test
        # fixtures too.
        from tokenizer.aligned_data.memmap_format import (
            encode_data_bin_prelude,
            encode_data_bin_trailer,
        )
        data_file.write(encode_data_bin_prelude())
        for spec in unmatched:
            typed_called = _project_called_typed(spec.called)
            registry.add(spec.func_name)
            for callee in spec.called:
                registry.add(callee)
            for version in spec.versions:
                offset, length = write_function_binary_data(
                    data_file, version.tokens, version.block_rl, version.insn_rl,
                    entry_idx=n_entries,
                )
                n_entries += 1
                unmatched_data_entries.append(
                    {
                        "func_name": spec.func_name,
                        "vkey": version.vkey,
                        "data_offset": offset,
                        "data_len": length,
                        "token_len": len(version.tokens),
                        "called": set(typed_called),
                        "extern_libraries": {},
                    }
                )
                lookup[(spec.func_name, version.vkey)] = (offset, length, 0)
        data_file.write(
            encode_data_bin_trailer(n_entries, cursor=data_file.tell())
        )
    return unmatched_data_entries, lookup
