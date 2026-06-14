"""Per-CSV vocab-era detection from token-stream type-coherence.

Single concern: given one per-binary tokenize CSV, decide which
reserved-prefix ERA its serialized vocab belongs to — legacy (content at
absolute id 256, ``value_negative`` absent: ``legacy_no_value_negative=
True``) versus modern (content at id 257, ``value_negative`` reserved at
slot 256: ``legacy_no_value_negative=False``). This is the only place
the vocab_unifier resolves era per-CSV; the loader's reconstruction
(``load_vocab_manager(..., legacy_no_value_negative=...)``) consumes the
result and nothing else crosses this boundary.

Why a detector and not a single global flag
--------------------------------------------
Eras #1 (legacy) and #3 (modern) are STRUCTURALLY INDISTINGUISHABLE in
the serialized vocab row (both omit ``value_negative``, both
``names[0]==block_v2``, both ``format_version=2``); the 256-vs-257
reserved count is applied at LOAD, not encoded in the row. A single
global flag therefore mis-resolves half of a MIXED-era corpus
(re-tokenized-subset@257 + untouched@256), shifting every token id→type
by 1 and corrupting the unified vocab + mapping + memmap. Re-tokenizing
only a bug-affected subset always produces a mixed corpus, so the era
must be resolved per-CSV.

The signal: carrier-band type-coherence
---------------------------------------
The two eras differ by exactly one inserted slot (``value_negative`` at
256). Resolving the same wire-stream content-marker id (``> 256``, the
carrier band per ``_inline_decode_state.real_mask``) through the VM
loaded at the WRONG offset is detectable in ONE direction only:

* MODERN data (era #3) loaded at offset 256 → the reconstructed
  ``id_to_token_type`` is one slot SHORTER, so the top carrier id falls
  OUT OF RANGE (and interior carriers may resolve to the reserved
  ``UNRESOLVED`` / ``VALUE_NEGATIVE`` band). Measurable, frequent.
* LEGACY data (era #1) loaded at offset 257 → the array is one slot
  LONGER; the shift is absorbed with ZERO out-of-range / reserved-band
  resolutions. The legacy era is therefore BLIND to carrier-band
  coherence — no per-CSV token-stream signal separates it from modern.
  (id-256's own resolution is offset-determined, not era-determined;
  per-position postfix grammar is fooled because a ``block_v2`` opener
  at id 256 mimics a valid ``value_negative`` postfix; frequency alone
  is not threshold-free because floatless/no-negative modern files
  exist.)

Consequence for the decision rule: the detector can POSITIVELY confirm
the MODERN era (clean at 257, invalid at 256) but cannot positively
confirm the legacy era. So it returns 257 (modern) only on positive
modern evidence and otherwise defers to the caller-supplied ``default``.
For a MIXED corpus the operator passes the legacy default (256); modern
files self-upgrade to 257 via positive detection while legacy + the
genuinely-modern-but-too-short-to-disambiguate files keep the default.
A UNIFORM modern corpus needs no special default: every CSV is
positively detected as 257.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec
from tokenizer.csv_files import open_versioned_csv_reader
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

from .loader import load_vocab_manager
from .types import Platform

logger = logging.getLogger(__name__)

# Absolute wire-stream layout constants (source of truth:
# VocabularyManager). The carrier band is ``id > _V2_VALUE_NEGATIVE_TOKEN_ID``
# (strict) — mirrors ``_inline_decode_state.real_mask``: id 256 is the
# sign slot (neither inline-digit nor carrier), ids < 256 are inline-digit
# payloads, ids > 256 are the real content-marker carriers.
_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT          # 256
_SIGN_SLOT = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID         # 256

# Token column in the per-binary record row (see tokenizer.main_loop's
# header: function_name, occurrence, tokens_base64, ...). Record rows are
# exactly this many columns; vocab-def rows (interspersed every 16384
# functions + the trailing one) are wider, so the column count alone
# discriminates record rows from vocab-def rows.
_TOKENS_COLUMN = 2
_RECORD_COLUMN_COUNT = 6

# Reserved-band token types a CARRIER (id > 256) must never resolve to.
# A carrier resolving here (or out of range) is a structurally-impossible
# resolution — the WRONG-offset fingerprint.
_RESERVED_TYPES = frozenset((TokenType.UNRESOLVED, TokenType.VALUE_NEGATIVE))

# Default cap on how many record rows we decode for the carrier sample.
# A few hundred functions exercise the full carrier id range densely
# enough to surface the top-id out-of-range signal; reading the whole
# CSV would be wasteful on large binaries.
_DEFAULT_SAMPLE_RECORD_LIMIT = 512


def _sample_carrier_ids(
    csv_path: Path,
    *,
    sample_record_limit: int,
) -> np.ndarray:
    """Decode up to ``sample_record_limit`` record rows' token streams and
    return the concatenated CARRIER ids (wire ids ``> 256``).

    Record rows are identified positionally (exactly
    ``_RECORD_COLUMN_COUNT`` columns); the wider vocab-def rows
    interspersed by the tokenizer's periodic ``save_vocabulary`` calls —
    and the trailing vocab-def line — are skipped by the column-count
    test. Returns an empty array when the CSV has no decodable records.
    """
    carrier_chunks: list[np.ndarray] = []
    with open(csv_path, newline="") as csvfile:
        reader, _format_version = open_versioned_csv_reader(csvfile)
        # First record-shaped row is the header (function_name, ...); its
        # tokens cell is the literal string "tokens_base64", not base64,
        # so it fails to decode — guarded by the try/except below.
        seen = 0
        for row in reader:
            if seen >= sample_record_limit:
                break
            if len(row) != _RECORD_COLUMN_COUNT:
                continue  # vocab-def row or malformed — not a record
            cell = row[_TOKENS_COLUMN]
            try:
                tokens = base64_to_ndarray_vec(cell)
            except Exception:
                # Header row or a corrupt cell — skip without counting it
                # against the sample budget so we still reach real records.
                continue
            seen += 1
            carriers = tokens[tokens > _SIGN_SLOT]
            if carriers.size:
                carrier_chunks.append(carriers.astype(np.int64))
    if not carrier_chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(carrier_chunks)


def _count_carrier_invalids(
    vocab_manager: VocabularyManager,
    carrier_ids: np.ndarray,
) -> int:
    """Count carrier ids that resolve INCOHERENTLY through ``vocab_manager``:
    out of range, or resolving to a reserved-band token type
    (``UNRESOLVED`` / ``VALUE_NEGATIVE``) where a content carrier must be.
    """
    id_to_token_type = vocab_manager.id_to_token_type
    n = len(id_to_token_type)
    if carrier_ids.size == 0:
        return 0
    out_of_range = carrier_ids >= n
    in_range_ids = carrier_ids[~out_of_range]
    invalid = int(out_of_range.sum())
    if in_range_ids.size:
        resolved = id_to_token_type[in_range_ids]
        for reserved in _RESERVED_TYPES:
            invalid += int(np.count_nonzero(resolved == reserved))
    return invalid


def detect_legacy_no_value_negative(
    csv_path: Path,
    *,
    default: bool,
    sample_record_limit: int = _DEFAULT_SAMPLE_RECORD_LIMIT,
) -> bool:
    """Detect the ``legacy_no_value_negative`` loader flag for one
    per-binary CSV from its token-stream carrier-band type-coherence.

    Returns ``False`` (modern / offset-257) ONLY on POSITIVE modern
    evidence: the carrier sample resolves with zero invalids at offset
    257 AND with at least one invalid at offset 256 (the modern-misloaded
    fingerprint). In every other case — legacy data (carrier-blind), a
    degenerate / too-short CSV with no disambiguating carriers, or an
    unloadable CSV — it returns ``default`` unchanged.

    ``default`` is the caller-supplied era (the unifier threads its
    ``insert_value_negative`` flag here): ``True`` = legacy/256,
    ``False`` = modern/257. The detection never overrides a confident
    modern call, but a confident modern call always wins over ``default``.
    """
    vm_legacy = load_vocab_manager(csv_path, legacy_no_value_negative=True)
    vm_modern = load_vocab_manager(csv_path, legacy_no_value_negative=False)
    if vm_legacy is None or vm_modern is None:
        # No loadable vocab-def — the unifier's own load step will skip
        # the file; defer the era to the caller default so this detector
        # never invents a value for a file that won't be loaded anyway.
        return default

    carrier_ids = _sample_carrier_ids(
        csv_path, sample_record_limit=sample_record_limit
    )
    if carrier_ids.size == 0:
        return default

    invalid_legacy = _count_carrier_invalids(vm_legacy, carrier_ids)
    invalid_modern = _count_carrier_invalids(vm_modern, carrier_ids)

    # Positive modern evidence: modern offset resolves cleanly AND the
    # legacy offset is demonstrably wrong (the modern-as-256 mis-load
    # shortens the array → out-of-range / reserved-band hits). Legacy
    # data resolves cleanly under BOTH offsets, so it never trips this
    # branch and correctly falls through to ``default``.
    if invalid_modern == 0 and invalid_legacy > 0:
        return False  # modern / offset-257

    return default
