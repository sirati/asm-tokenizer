"""Shared synthetic-binary fixture for the BinarySession test suite.

Single concern: lay down a one-matched + one-unmatched corpus with a
real ``_variants.bin`` record at byte 0, and assemble the metadata bag
``BinarySession`` consumes. Split out so the two session test files
(``test_session`` lifecycle + ``test_session_exception_safety``) can
share one ``synthetic_binary`` fixture without duplicating the wiring.

The fixture pairs the corpus builder (``_corpus.build_corpus_with_registry``)
with a caller-supplied variant registry so every section-row
``variant_ref`` cell points at the single hand-laid bin record. The
variants bin is hand-laid because the resolver needs a real
``encode_record``-produced byte sequence at the offset the rows
reference; the stub builder's default registry emits placeholders that
wouldn't decode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.index_format import read_index_arrays
from tokenizer.aligned_data.loader._sections_bin_walk import (
    unmatched_region_start,
)

from ._corpus import (
    MatchedFunctionSpec,
    UnmatchedFunctionSpec,
    build_corpus_with_registry,
    make_simple_variant,
)


class _FakeArm:
    """Minimal SectionArm stand-in -- session reads attributes only.

    Tests build arms directly from on-disk byte positions rather than
    going through ``load_section_arm`` so the lifecycle coverage stays
    independent of the parallel matched-arm reader rewrite. Records
    are self-describing in ``_data.bin`` so the fake carries no
    ``lengths`` companion.
    """

    def __init__(
        self,
        starts: np.ndarray,
        func_names: List[str] | None = None,
        section_starts: np.ndarray | None = None,
        bin_starts: np.ndarray | None = None,
        bin_lengths: np.ndarray | None = None,
        record_to_section_idx: np.ndarray | None = None,
    ) -> None:
        self.starts = starts
        self.func_names = func_names or []
        self.section_starts = section_starts
        # Matched arm: session.load_matched(idx) slices ``sections.bin``
        # via these per-function fields (per the Phase 4 SectionArm
        # shape); ``starts`` carries per-variant data-bin offsets for
        # the validator. Unmatched arm leaves these as None and reuses
        # ``starts`` directly.
        self.bin_starts = bin_starts
        self.bin_lengths = bin_lengths
        # Per-record -> per-section mapping; the unmatched session
        # path looks it up to resolve owning sections in O(1).
        self.record_to_section_idx = record_to_section_idx


class _FakeVocab:
    """Vocab stub: just enough for the variant decoder + axis builder.

    Token IDs start at the v2 eager-block end (272) so axis tokens land
    in the metadata-variant band, matching the canonical layout (digits
    0..255, value_negative 256, NUMBER+IDENTITY 257..271, instruction
    reps + tail metadata variants >= 272). Starting at 256 would alias
    the value_negative sign marker, which breaks any test that walks
    ``full_token_stream`` through :func:`build_inline_decode_state`
    (which asserts the leading position is not in the value band).
    """

    def __init__(self, items: List[str]) -> None:
        self._s2i = {s: i + 272 for i, s in enumerate(items)}
        self._i2s = {v: k for k, v in self._s2i.items()}
        # Real vocabs publish ``value_negative_token_id`` directly. ``None``
        # mirrors the pre-Phase-4 default that real fakes inherited via the
        # old scan-based resolver — tests that need value_negative-aware
        # decoding can override this on the instance.
        self.value_negative_token_id: int | None = None
        # The unified-vocab gate (format_version=1) is the only one the
        # v2 decode contract supports. Tests that need a different value
        # can override on the instance.
        self.format_version: int = 1

    def get_token_id(self, token: str) -> int:
        return self._s2i.get(token, -1)

    def get_token_str(self, token_id: int) -> str:
        return self._i2s.get(token_id, "")


def write_variants_slim_csv(
    base: Path,
    binary_name: str,
    offset_to_filename: Dict[int, str],
) -> None:
    """Lay down the slim ``_variants.csv`` that pairs ``_variants.bin``.

    Mirrors the production builder's atomic pair-write (memmap_builder/
    variants.py): a ``# format=N`` prelude, a
    ``filename,variant_id,offset`` header, and one row per bin record
    (offset as bare hex). Hand-laid fixtures that write ``_variants.bin``
    MUST call this too -- ``BinaryDataset`` hard-fails on a half-present
    pair, and the resolver needs a filename for every offset it reads.
    """
    import csv as _csv

    from tokenizer.aligned_data.csv_format import write_csv_prelude

    csv_path = base / f"{binary_name}_variants.csv"
    with open(csv_path, "w", newline="", encoding="ascii") as handle:
        write_csv_prelude(handle)
        writer = _csv.writer(handle, lineterminator="\n")
        writer.writerow(["filename", "variant_id", "offset"])
        for vid, (offset, filename) in enumerate(sorted(offset_to_filename.items())):
            writer.writerow([filename, f"{vid:08x}", f"{offset:x}"])


def _write_variants_bin(base: Path, binary_name: str, vocab: _FakeVocab) -> int:
    """Lay down the ``_variants.bin`` + slim ``_variants.csv`` pair.

    Returns the byte offset of the single record. Hand-laid (not via the
    fixture builder) because the variant resolver needs a real axis
    record produced by ``encode_record`` to round-trip back through the
    decoder. The slim CSV is written alongside so the corpus passes
    ``BinaryDataset``'s sidecar-pair invariant.
    """
    from tokenizer.variant_tokens.encoder import encode_record

    class _V:
        arch = "x86_64"
        compiler = "gcc"
        compilerversion = "13.2.0"
        opt = "-O2"
        extra_metadata: Dict[str, Any] = {}

    record = encode_record(_V(), vocab)
    variants_path = base / f"{binary_name}_variants.bin"
    with open(variants_path, "wb") as f:
        offset = f.tell()
        f.write(record.tobytes())
    write_variants_slim_csv(
        base, binary_name, {offset: f"{binary_name}-x86_64-gcc-13.2.0-O2"}
    )
    return offset


class _VariantStubRegistry:
    """Registry whose ``.ref(vkey)`` returns the supplied ``offset_hex``.

    The integer companion ``.byte_offset(vkey)`` (consumed by the
    matched-sections BIN walker) parses the same hex string so the
    CSV's ``variant_ref`` cell and the BIN's ``variant_ref_offset``
    u32 agree on the per-vkey value.
    """

    def __init__(self, hex_for_vkey: dict) -> None:
        self._hex = hex_for_vkey

    def ref(self, vkey) -> str:
        return self._hex[vkey]

    def byte_offset(self, vkey) -> int:
        return int(self._hex[vkey], 16)


def build_synthetic_binary(tmp_path: Path) -> Dict[str, Any]:
    """Lay down a tiny binary: one matched section + one unmatched.

    Matched arm: ``my_func`` with two variants (pass-1 dedupe heuristic
    requires distinct data offsets). Unmatched arm: ``lonely_func``
    with one version. All variant_refs point at the single
    ``_variants.bin`` record so the resolver round-trip exercises the
    axis decoder.
    """
    base = tmp_path
    binary_name = "tinybin"

    vocab_strings = [
        "arch:x64",
        "comp:gcc",
        "cver:gcc:13.2.0",
        "opt:O2",
    ]
    vocab = _FakeVocab(vocab_strings)
    variant_offset = _write_variants_bin(base, binary_name, vocab)
    variant_ref_hex = f"{variant_offset:x}"

    m_vkey_a = ("matched", 0)
    m_vkey_b = ("matched", 1)
    u_vkey = ("unmatched", 0)
    matched_specs = (
        MatchedFunctionSpec(
            func_name="my_func",
            variants=(
                make_simple_variant(m_vkey_a, token_seed=1, n_tokens=8),
                make_simple_variant(m_vkey_b, token_seed=2, n_tokens=6),
            ),
            called=(),
        ),
    )
    unmatched_specs = (
        UnmatchedFunctionSpec(
            func_name="lonely_func",
            versions=(make_simple_variant(u_vkey, token_seed=3, n_tokens=4),),
            called=(),
        ),
    )

    variants_registry = _VariantStubRegistry(
        {m_vkey_a: variant_ref_hex,
         m_vkey_b: variant_ref_hex,
         u_vkey: variant_ref_hex}
    )
    corpus = build_corpus_with_registry(
        base, binary_name,
        matched=matched_specs, unmatched=unmatched_specs,
        variants=variants_registry,
    )

    matched_arm = _matched_arm_from_corpus(corpus)
    unmatched_arm = _unmatched_arm_from_corpus(corpus)

    from tokenizer.aligned_data.loader.function_names_loader import (
        load_function_names,
    )
    from tokenizer.aligned_data.loader.extern_providers_loader import (
        load_extern_providers,
    )
    _, line_to_name = load_function_names(corpus.function_names_sidecar)
    line_to_provider = load_extern_providers(corpus.extern_providers_sidecar)

    metadata = {
        "matched_arm": matched_arm,
        "unmatched_arm": unmatched_arm,
        "offset_to_filename": {variant_offset: "tinybin-x64-gcc-13.2.0-O2"},
        "line_to_name": line_to_name,
        "line_to_provider": line_to_provider,
    }
    return {
        "base_path": base,
        "binary_name": binary_name,
        "vocab": vocab,
        "metadata": metadata,
        "variant_offset": variant_offset,
    }


def _matched_arm_from_corpus(corpus) -> _FakeArm:
    pair = read_csv_section_index_arrays(corpus.matched_index_bin)
    assert pair is not None and pair[0].shape == (1,)
    bin_starts, bin_lengths = pair
    # arm.starts is only read by validators / iterators (not by
    # load_matched after the Phase 4 contract change); populate an
    # empty placeholder so any inadvertent slice surfaces clearly.
    empty = np.zeros(0, dtype=np.int64)
    return _FakeArm(
        starts=empty,
        func_names=["my_func"],
        bin_starts=bin_starts,
        bin_lengths=bin_lengths,
    )


def _unmatched_arm_from_corpus(corpus) -> _FakeArm:
    starts = read_index_arrays(corpus.unmatched_index_bin)
    assert starts is not None
    # Unmatched section in the BIN sits right after the matched-arm
    # region; reuse the loader's own offset helper instead of mirroring
    # its logic in the fixture.
    unmatched_section_offset = unmatched_region_start(corpus.matched_index_bin)
    # Synthetic fixture: one unmatched function -> one section -> one
    # record. Pin the per-record -> per-section mapping to that single
    # slot so the session's O(1) dispatch finds the owning section.
    record_to_section_idx = np.zeros(len(starts), dtype=np.uint32)
    return _FakeArm(
        starts=starts,
        func_names=["lonely_func"],
        section_starts=np.array([unmatched_section_offset], dtype=np.int64),
        record_to_section_idx=record_to_section_idx,
    )


def count_open_fds() -> int:
    """Count open file descriptors for the current process (Linux only)."""
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except FileNotFoundError:  # pragma: no cover -- non-Linux fallback
        return -1


@pytest.fixture
def synthetic_binary(tmp_path: Path) -> Dict[str, Any]:
    """Pytest wrapper around :func:`build_synthetic_binary`."""
    return build_synthetic_binary(tmp_path)
