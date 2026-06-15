"""Pure encode/decode of variant-axis token records against a vocab.

Single concern: convert one variant identity into its bin-record
``np.ndarray[uint16]`` (and back). The layout is::

    +-----+-----+-----+-----+-----+-----+----- ... -----+
    | u16 | u16 | u16 | u16 | u16 | u16 ...              |
    | n   |arch | comp| cver| opt | metadata k/v tokens  |
    +-----+-----+-----+-----+-----+-----+----- ... -----+

``n`` is the count of axis tokens (not the array length): the returned
array has length ``1 + n``. Total on-disk bytes are ``2 + 2*n`` —
matches the size header convention used by ``record.read_record``.

This module is intentionally agnostic of file handles (that is
``record.py``'s concern) and of token registration (that is the
unifier's concern in Batch 3 — ``encode_record`` only LOOKS UP).
Missing lookups indicate a discovery/registration bug upstream and
fail loudly via ``AssertionError``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

from .prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    N_POSITIONAL_AXES,
    OPT_PREFIX,
    build_axis_strings,
)

# uint16 wire-format ceiling. Matches ``_data.bin``'s existing uint16
# layout — every variant token ID must fit so the decoder's
# ``np.frombuffer(..., dtype=np.uint16)`` round-trip is lossless. A
# silent truncation to 16 bits would corrupt the dataset, so the
# encoder hard-asserts.
_UINT16_MAX = 0xFFFF


def _lookup_id(vocab: Any, token: str) -> int:
    """Resolve token string to vocab ID, hard-failing on miss.

    The unifier's pass 1 (Batch 3) registers every variant-axis token
    string into the unified VM before the encoder runs. A missing
    lookup here means discovery missed a variant — that is a bug
    upstream, not something the encoder should silently coerce.
    """
    token_id = vocab.get_token_id(token)
    # Explicit raises, NOT assert: this is the last-line guard against
    # writing a silently-wrong id into the variants bin, and `python -O`
    # strips asserts -- which would turn a phase-2-missed-token into a
    # ``-1`` (-> 0xFFFF OOB id) and a vocab-overgrows-uint16 into a silent
    # truncation, in exactly the optimised production run where it matters.
    if token_id == -1:
        raise ValueError(
            f"variant-axis token {token!r} not registered in unified vocab; "
            "upstream discovery (VariantInventory + register pass) missed it"
        )
    if not (0 <= token_id <= _UINT16_MAX):
        raise ValueError(
            f"variant-axis token {token!r} got vocab id {token_id} which "
            f"exceeds uint16 ceiling ({_UINT16_MAX}); the unified vocab has "
            "outgrown the wire format and the bin layout must bump"
        )
    return token_id


def encode_record(version_info: Any, vocab: Any) -> npt.NDArray[np.uint16]:
    """Encode one variant's axis tokens into the ``[n, *ids]`` layout."""
    token_strings: List[str] = build_axis_strings(version_info)
    n_tokens = len(token_strings)
    # Defensive: there must be exactly 4 positional axes + tail. The
    # tail count comes from ``build_metadata_tokens`` so this is a
    # ``build_axis_strings`` contract check, not a user-facing
    # validator.
    assert n_tokens >= N_POSITIONAL_AXES, (
        f"build_axis_strings returned {n_tokens} tokens, expected at "
        f"least {N_POSITIONAL_AXES} positional axes"
    )
    # Header u16 must also fit; n_tokens > 65535 means a single variant
    # carries >65k metadata pairs, which would already have blown the
    # vocab.
    assert n_tokens <= _UINT16_MAX, (
        f"variant record has {n_tokens} tokens, exceeds uint16 size "
        f"header ceiling ({_UINT16_MAX})"
    )
    out = np.empty(1 + n_tokens, dtype=np.uint16)
    out[0] = n_tokens
    for i, token in enumerate(token_strings, start=1):
        out[i] = _lookup_id(vocab, token)
    return out


def decode_record(tokens: npt.NDArray[np.uint16], vocab: Any) -> Dict[str, Any]:
    """Decode a ``[n, *ids]`` array back into the per-axis dict.

    The returned dict shape is::

        {
            "arch": str,
            "compiler": str,
            "compilerversion": str,
            "opt": str,
            <metakey>: [val1, val2, ...],   # always list (plan §
            ...                              # "always-list metadata")
        }

    Metadata keys are recovered by splitting each tail token on its
    FIRST ``:`` — the ``inventory.add()`` invariant guarantees the
    pre-colon part is the real key. Single-value keys decode to a
    length-1 list for schema regularity.

    Note: this decoder reverses the prefix grammar in ``prefixes.py``;
    the ``arch`` value comes back already alias-collapsed (e.g.
    ``x64``) — the original sidecar ``arch`` string (e.g. ``amd64``)
    is not recoverable. That is by design: alias collapse is the whole
    point of ``arch_to_variant_arch``.
    """
    assert tokens.dtype == np.uint16, (
        f"decode_record expects uint16 input, got {tokens.dtype}"
    )
    n_tokens = int(tokens[0])
    assert n_tokens >= N_POSITIONAL_AXES, (
        f"variant record header n={n_tokens} below {N_POSITIONAL_AXES} "
        "positional axes; record is corrupt or truncated"
    )
    assert len(tokens) >= 1 + n_tokens, (
        f"variant record array length {len(tokens)} smaller than header "
        f"requires (1 + {n_tokens})"
    )

    # Positional axes — strip prefix from each.
    arch_str = _strip_prefix(vocab.get_token_str(int(tokens[1])), ARCH_PREFIX)
    comp_str = _strip_prefix(vocab.get_token_str(int(tokens[2])), COMP_PREFIX)
    cver_raw = _strip_prefix(vocab.get_token_str(int(tokens[3])), CVER_PREFIX)
    # cver string is ``<compiler>:<version>`` — split off the compiler
    # we already decoded so the remainder is the version.
    cver_prefix = f"{comp_str}:"
    assert cver_raw.startswith(cver_prefix), (
        f"cver token {cver_raw!r} does not start with decoded compiler "
        f"{comp_str!r}; record is corrupt or compiler/version were "
        "registered with a different prefix grammar"
    )
    version = cver_raw[len(cver_prefix):]
    opt = _strip_prefix(vocab.get_token_str(int(tokens[4])), OPT_PREFIX)

    out: Dict[str, Any] = {
        "arch": arch_str,
        "compiler": comp_str,
        "compilerversion": version,
        "opt": opt,
    }

    # Metadata tail — split on first ":" only so values may contain
    # colons (e.g. URL-like metavalues) without ambiguity.
    metadata: Dict[str, List[str]] = {}
    for i in range(1 + N_POSITIONAL_AXES, 1 + n_tokens):
        meta_token = vocab.get_token_str(int(tokens[i]))
        key, sep, value = meta_token.partition(":")
        assert sep == ":", (
            f"metadata token {meta_token!r} has no ':' separator; "
            "record / vocab inconsistency"
        )
        metadata.setdefault(key, []).append(value)

    # Sort each value list deterministically so a re-encode of the
    # result yields a record byte-identical to the input (round-trip
    # property exercised by the test suite).
    for key in metadata:
        metadata[key].sort()
    out.update(metadata)
    return out


def _strip_prefix(token: str, prefix: str) -> str:
    assert token.startswith(prefix), (
        f"vocab token {token!r} does not start with expected prefix "
        f"{prefix!r}; positional-axis ordering is wrong or vocab is "
        "corrupt"
    )
    return token[len(prefix):]
