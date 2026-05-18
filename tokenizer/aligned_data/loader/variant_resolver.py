"""Variant-ref resolution against a session's open ``_variants.bin``.

Single concern: take a hex-string ``variant_ref`` (a byte offset into
``<bin>_variants.bin``) and a session's open handles, return the
fully-decoded variant identity dict the dataloader hands to callers.

Pure-function module — every input is explicit so the same body works
from a ``BinarySession`` (long-lived handles) and from any future
notebook / script caller that wants to resolve one ref without a
session wrapper. Nothing here holds state.

Round-trip ownership:
    - The slim ``<bin>_variants.csv`` (header ``filename,offset``) is
      read once per ``BinaryDataset`` via
      ``load_variants_offset_to_filename`` and cached as a
      ``dict[int, str]`` keyed by integer offset.
    - The open ``_variants.bin`` memmap belongs to the session.
    - The unified ``VocabularyManager`` belongs to the
      ``AlignedDataLoader`` (loaded once, shared across all sessions).
    - ``get_variant_by_ref`` consumes all three plus a hex ``ref``.

Returned dict shape — always-list metadata + filename + raw tokens::

    {
        "arch": str,
        "compiler": str,
        "compilerversion": str,
        "opt": str,
        <metakey>: [val1, val2, ...],   # always list per plan
        ...
        "filename": str,
        "variant_tokens": np.ndarray[uint16],
    }

``variant_tokens`` carries the vocab IDs ONLY (the leading ``n_tokens``
size header from ``record.read_record`` is dropped). Rationale:
``FunctionData.full_token_stream()`` concatenates this array with the
instruction token stream — the header is record-layout metadata, not a
vocab ID, so leaving it in would inject a meaningless ID into the
stream.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np
import numpy.typing as npt

from tokenizer.variant_tokens.encoder import decode_record
from tokenizer.variant_tokens.record import read_record


def load_variants_offset_to_filename(slim_csv_path: Path) -> Dict[int, str]:
    """Read the slim ``<bin>_variants.csv`` into ``{offset: filename}``.

    Slim CSV header is ``filename,offset`` where ``offset`` is the hex
    byte offset (no ``0x`` prefix, matching ``f"{offset:x}"``) into the
    sibling ``_variants.bin``. Missing sidecar yields an empty dict —
    legacy datasets predate this schema; callers see "no resolvable
    refs" rather than a crash.

    The dict is keyed by ``int`` (not the hex string) because callers
    already parse ``int(ref, 16)`` once for the memmap slice; reusing
    that integer for the filename lookup avoids a redundant conversion
    on the hot path.
    """
    if not slim_csv_path.exists():
        return {}
    out: Dict[int, str] = {}
    with open(slim_csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["offset"], 16)] = row["filename"]
    return out


def get_variant_by_ref(
    ref: str,
    vocab_manager: Any,
    variants_mmap: npt.NDArray[np.uint8],
    offset_to_filename: Dict[int, str],
) -> Dict[str, Any]:
    """Resolve one ``variant_ref`` to its full identity dict.

    Steps (matches plan §"get_variant_by_ref (in-session)"):
        1. Parse ``ref`` as hex → byte offset into ``variants_mmap``.
        2. ``record.read_record`` slices ``[n_tokens, *ids]`` (uint16).
        3. ``encoder.decode_record`` resolves each ID through
           ``vocab_manager`` to ``{arch, compiler, compilerversion,
           opt, <metakey>: [vals]}``.
        4. ``offset_to_filename`` supplies the human-readable filename.
        5. ``variant_tokens`` is the vocab-ID slice ``tokens[1:]`` —
           the leading size header is dropped so the array can be
           concatenated directly into the instruction token stream.
    """
    offset = int(ref, 16)
    tokens = read_record(variants_mmap, offset)
    out: Dict[str, Any] = decode_record(tokens, vocab_manager)
    out["filename"] = offset_to_filename[offset]
    out["variant_tokens"] = tokens[1:]
    return out
