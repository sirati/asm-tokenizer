"""Rich-body corpus fixture for the dense-sidecar byte-identity gate.

Single concern: lay down a memmap corpus whose decoded bodies carry REAL
NUMBER + IDENTITY + COUNTER carriers (with inline-digit payloads) AND a
genuinely-inlined splice (root -> leaf via a SHARED vkey namespace +
per-variant call sets), so the dense identity / numeric sidecars are
NON-TRIVIAL: the number decode emits significands, the FUNCTION dedup
remap mints cross-call_target counters, and the COUNTER offset bump fires.

The decode-agnostic ``sorted_index`` fixtures (``make_simple_variant``)
deliberately emit instruction-rep-band tokens only (no carriers), so the
dense arrays they produce are trivially empty -- a vacuous gate. This
fixture exists ONLY to make the dense byte-identity assertions
load-bearing.

Wire form (unified vocab, all ids from :class:`VocabularyManager`):

* inline-digit bytes: ``[0, _V2_RESERVED_DIGIT_COUNT)`` -- a carrier's
  big-endian payload, MSB-first.
* sign marker: ``_V2_VALUE_NEGATIVE_TOKEN_ID`` -- optional postfix on a
  NUMBER carrier (makes the source negative).
* NUMBER carriers: ``[_V2_NUMBER_BLOCK_START, +COUNT)`` -- VC2 first.
* IDENTITY carriers: ``[_V2_IDENTITY_BLOCK_START, _V2_EAGER_BLOCK_END)``
  -- BLOCK (COUNTER), then LOCAL_FUNC / PLT_FUNC (FUNCTION), ...
* instruction-rep: ``>= _V2_EAGER_BLOCK_END``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
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

# Identity-block offsets (canonical layout; see ``_dedup_walk._constants``).
_BLOCK = _IDENTITY_START + 0  # COUNTER category
_LOCAL_FUNC = _IDENTITY_START + 1  # FUNCTION category
_VC2 = _NUMBER_START  # first NUMBER carrier
_FLOAT16 = _NUMBER_START + 1


def _rich_body(*, instr_base: int, with_negative: bool) -> np.ndarray:
    """A v2 wire body exercising every dense-decode arm.

    Layout (carriers + payloads, then an instruction-rep token):

    * a BLOCK COUNTER carrier with a 1-byte payload (caller-local id) ->
      ALG-4 offset bump arm.
    * a LOCAL_FUNC carrier with caller-local id ``0`` (the root's first
      LOCAL call target = the leaf) -> ALG-3 FUNCTION dedup arm.
    * a VC2 NUMBER carrier with a 2-byte payload (optionally negated) ->
      VC2 significand + sign arm.
    * an F16 NUMBER carrier with a 2-byte payload -> IEEE-narrow arm.
    * an instruction-rep token (band ``>= _EAGER_END``) so the stream is
      not pure-carriers.
    """
    pieces: list[int] = []
    # BLOCK COUNTER carrier, caller-local id 3 (1-byte payload).
    pieces += [_BLOCK, 3]
    # LOCAL_FUNC carrier, caller-local id 0 (the leaf call target).
    pieces += [_LOCAL_FUNC, 0]
    # VC2 carrier, 2-byte big-endian payload 0x0102, optional sign marker.
    pieces += [_VC2, 0x01, 0x02]
    if with_negative:
        pieces += [_SIGN]
    # F16 carrier, 2-byte payload 0x3C00 (= 1.0).
    pieces += [_FLOAT16, 0x3C, 0x00]
    # An instruction-rep token.
    pieces += [instr_base]
    return np.asarray(pieces, dtype=np.uint16)


def _leaf_body(*, instr_base: int) -> np.ndarray:
    """A leaf body carrying its own carriers (so the inlined callee adds
    fresh dense entries)."""
    pieces: list[int] = []
    # A second BLOCK COUNTER carrier (caller-local id 1) -> the offset
    # bump must shift it past the root's BLOCK count.
    pieces += [_BLOCK, 1]
    # A VC2 carrier with a different 2-byte payload.
    pieces += [_VC2, 0x00, 0x05]
    pieces += [instr_base]
    return np.asarray(pieces, dtype=np.uint16)


def _spec_variant(vkey, tokens, *, called=None) -> VariantSpec:
    block_rl = np.array([len(tokens)], dtype=np.uint8)
    insn_rl = np.array([len(tokens)], dtype=np.uint8)
    return VariantSpec(
        vkey=vkey, tokens=tokens, block_rl=block_rl, insn_rl=insn_rl,
        called=called,
    )


def build_rich_splice_fixture(tmp_path: Path) -> Path:
    """Corpus: ``root`` (2 variants) inlines ``leaf`` (shared vkeys).

    ``root`` variant 0 calls ``leaf``; variant 1 does not -- so the
    once-only / all-variants-equivalence inclusion inlines ``leaf`` on
    variant 0's row (a real multi-call_target row). Both functions carry
    rich carrier bodies, and a ``solo`` companion gives the catalog a
    second section. The SHARED ``("V", i)`` vkey namespace makes the
    caller's per-call entry resolve to the leaf's matching variant (a
    real inline, NOT a MISSING_VARIANT_INDEX slot).
    """
    base = tmp_path / "rich_splice"
    base.mkdir(parents=True, exist_ok=True)

    root_variants = (
        _spec_variant(
            ("V", 0),
            _rich_body(instr_base=_EAGER_END + 5, with_negative=True),
            called=("leaf",),
        ),
        _spec_variant(
            ("V", 1),
            _rich_body(instr_base=_EAGER_END + 9, with_negative=False),
            called=(),
        ),
    )
    leaf_variants = (
        _spec_variant(("V", 0), _leaf_body(instr_base=_EAGER_END + 13)),
        _spec_variant(("V", 1), _leaf_body(instr_base=_EAGER_END + 17)),
    )
    solo_variants = (
        make_simple_variant(("solo", 0), token_seed=50, n_tokens=6),
    )
    matched = (
        MatchedFunctionSpec(
            func_name="root", variants=root_variants, called=("leaf",)
        ),
        MatchedFunctionSpec(
            func_name="leaf", variants=leaf_variants, called=()
        ),
        MatchedFunctionSpec(
            func_name="solo", variants=solo_variants, called=()
        ),
    )
    build_corpus_with_registry(
        base,
        _BINARY_NAME,
        matched=matched,
        unmatched=(),
        variants=_DeterministicVariantRegistry(),
    )
    return base
