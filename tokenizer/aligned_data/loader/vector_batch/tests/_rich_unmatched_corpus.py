"""Rich-body UNMATCHED-arm corpus for the unmatched byte-identity gate.

Single concern: lay down a memmap corpus whose UNMATCHED arm carries the
two unmatched-root behaviours the byte-identity gate must pin:

* a MULTI-version unmatched root whose versions call DIFFERENT callee
  sets (FLAG-A: a callee reached by some-but-not-all versions inlines);
* an unmatched root that calls BOTH an unmatched callee (a real inline)
  AND a matched callee (DROPPED cross-arm -- the unmatched arm's
  ``LiveNodeAdjacency`` ``_sec_map`` holds only unmatched section
  offsets, so the matched callee misses -> -1 -> dropped, exactly
  ``batch_decode``'s arm-keyed behaviour).

Every unmatched root / callee body carries REAL NUMBER + IDENTITY + FID
carriers (mirroring :mod:`._rich_corpus`), so the dense byte-identity
assertions are NON-vacuous.

The unmatched functions are laid down so the cross-arm-drop root's BASE
RECORD idx is SHIFTED past its columnar SECTION idx: a leading 2-version
``pad`` function consumes record slots ahead of it, so ``base_record_idx
!= section_idx``. This exercises the record-idx -> section-idx mapping in
:func:`...vector_batch._entry._columnar_section_idx` (the byte-offset
lookup that must NOT assume ``pointer.idx == columnar section idx`` for
the unmatched arm).

Wire form (unified vocab, ids from :class:`VocabularyManager`) -- identical
band conventions to :mod:`._rich_corpus`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    UnmatchedFunctionSpec,
    build_corpus_with_registry,
)
from tokenizer.aligned_data.loader.tests._corpus.specs import (
    VariantSpec,
    make_simple_variant,
)
from tokenizer.token_manager import VocabularyManager

from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    _DeterministicVariantRegistry,
)


_BINARY_NAME = "sortbin"

_DIGIT = VocabularyManager._V2_RESERVED_DIGIT_COUNT
_SIGN = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID
_NUMBER_START = VocabularyManager._V2_NUMBER_BLOCK_START
_IDENTITY_START = VocabularyManager._V2_IDENTITY_BLOCK_START
_EAGER_END = VocabularyManager._V2_EAGER_BLOCK_END

_BLOCK = _IDENTITY_START + 0  # COUNTER category
_LOCAL_FUNC = _IDENTITY_START + 1  # FUNCTION category
_VC2 = _NUMBER_START  # first NUMBER carrier
_FLOAT16 = _NUMBER_START + 1


def _root_body(*, instr_base: int, call_slots, with_negative: bool) -> np.ndarray:
    """A v2 wire body carrying ``call_slots`` LOCAL_FUNC calls + carriers.

    ``call_slots`` is the per-version sequence of caller-local call_target
    ids (each ``[_LOCAL_FUNC, slot]``). The id is the index into the
    function's union ``called`` tuple -- slot 0 = first callee, etc. Plus
    a BLOCK COUNTER carrier, a VC2 + F16 NUMBER carrier, and a trailing
    instruction-rep token (so the dense decode is non-vacuous).
    """
    pieces: list[int] = []
    # BLOCK COUNTER carrier, caller-local id 3 (1-byte payload).
    pieces += [_BLOCK, 3]
    for slot in call_slots:
        pieces += [_LOCAL_FUNC, slot]
    # VC2 carrier, 2-byte big-endian payload 0x0102, optional sign marker.
    pieces += [_VC2, 0x01, 0x02]
    if with_negative:
        pieces += [_SIGN]
    # F16 carrier, 2-byte payload 0x3C00 (= 1.0).
    pieces += [_FLOAT16, 0x3C, 0x00]
    pieces += [instr_base]
    return np.asarray(pieces, dtype=np.uint16)


def _leaf_body(*, instr_base: int) -> np.ndarray:
    """A leaf body with its OWN carriers (so an inlined callee adds dense
    entries that the offset bump + dedup remap must reconcile)."""
    pieces: list[int] = []
    pieces += [_BLOCK, 1]
    pieces += [_VC2, 0x00, 0x05]
    pieces += [instr_base]
    return np.asarray(pieces, dtype=np.uint16)


def _spec_version(vkey, tokens, *, called=None) -> VariantSpec:
    block_rl = np.array([len(tokens)], dtype=np.uint8)
    insn_rl = np.array([len(tokens)], dtype=np.uint8)
    return VariantSpec(
        vkey=vkey, tokens=tokens, block_rl=block_rl, insn_rl=insn_rl,
        called=called,
    )


def build_rich_unmatched_fixture(tmp_path: Path) -> Path:
    """Corpus whose UNMATCHED arm carries the two root behaviours.

    Unmatched functions, in record order:

    * ``upad`` -- 2 versions, no calls. PURPOSE: consume two record slots
      ahead of the roots so every later root's BASE RECORD idx is shifted
      past its columnar SECTION idx (``base_record_idx != section_idx``).
    * ``uroot`` -- 2 versions, FLAG-A: version 0 calls ``uleaf`` (slot 0,
      inlined) AND ``mcallee`` (slot 1, DROPPED cross-arm); version 1
      calls NEITHER. The columnwise-ALL exclusion thus differs across the
      two versions -> a real sampled-subset-vs-full inclusion divergence.
    * ``uleaf`` -- 2 versions, rich carriers, no calls. The inlined callee.

    Matched arm: ``mcallee`` -- the matched callee ``uroot`` v0 names; it
    resolves at build time (the merged ``function_lookup`` spans both
    arms) but lands in the MATCHED section map, so following it from the
    unmatched arm misses -> dropped (the cross-arm DROP the gate proves).
    """
    base = tmp_path / "rich_unmatched"
    base.mkdir(parents=True, exist_ok=True)

    upad_versions = (
        _spec_version(("upad", "u", 0), _leaf_body(instr_base=_EAGER_END + 21)),
        _spec_version(("upad", "u", 1), _leaf_body(instr_base=_EAGER_END + 25)),
    )
    # uroot + uleaf SHARE the ``("V", i)`` vkey namespace so the caller's
    # per-call entry resolves to the leaf's matching version (a REAL
    # inline, not a MISSING_VARIANT_INDEX miss) -- mirrors the matched
    # rich-splice fixture's shared-namespace convention.
    uroot_versions = (
        _spec_version(
            ("V", 0),
            _root_body(
                instr_base=_EAGER_END + 5,
                call_slots=(0, 1),  # 0 -> uleaf (inline), 1 -> mcallee (drop)
                with_negative=True,
            ),
            called=("uleaf", "mcallee"),
        ),
        _spec_version(
            ("V", 1),
            _root_body(
                instr_base=_EAGER_END + 9,
                call_slots=(),  # calls nothing -> FLAG-A divergence vs v0
                with_negative=False,
            ),
            called=(),
        ),
    )
    uleaf_versions = (
        _spec_version(("V", 0), _leaf_body(instr_base=_EAGER_END + 13)),
        _spec_version(("V", 1), _leaf_body(instr_base=_EAGER_END + 17)),
    )

    unmatched = (
        UnmatchedFunctionSpec(func_name="upad", versions=upad_versions, called=()),
        UnmatchedFunctionSpec(
            func_name="uroot",
            versions=uroot_versions,
            called=("uleaf", "mcallee"),
        ),
        UnmatchedFunctionSpec(func_name="uleaf", versions=uleaf_versions, called=()),
    )
    # ``mcallee`` shares the ``("V", i)`` namespace so a per-call entry
    # WOULD resolve to a matching version -- the ONLY reason it is dropped
    # is that it lives in the MATCHED arm (absent from the unmatched arm's
    # section map), isolating the cross-arm DROP from any vkey-miss.
    matched = (
        MatchedFunctionSpec(
            func_name="mcallee",
            variants=(
                make_simple_variant(("V", 0), token_seed=60, n_tokens=6),
                make_simple_variant(("V", 1), token_seed=61, n_tokens=6),
            ),
            called=(),
        ),
    )
    build_corpus_with_registry(
        base,
        _BINARY_NAME,
        matched=matched,
        unmatched=unmatched,
        variants=_DeterministicVariantRegistry(),
    )
    return base
