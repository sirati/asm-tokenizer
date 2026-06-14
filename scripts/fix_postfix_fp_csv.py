#!/usr/bin/env python3
"""Patch per-binary tokenizer-output CSVs in place: replace HALLUCINATED
postfix-FP float tokens with a value-less ``float_annotation`` token,
operating ONLY on the RAW uint16 token-id stream.

Background
----------
A prior encoder bug emitted, after an FP-typed ptr/const load, a bare
``floatXX`` token (one of the six float TYPE tokens) with NO inline value
("postfix FP annotation", documented in ``tokenizer/disasm/precedence.md``).
That broke decoding. asm-tokenizer is removing the encoder emission and
introducing a real ``float_annotation`` token. This script patches the
EXISTING corpus CSVs to swap each bare/broken ``floatXX`` for
``float_annotation`` (1 token -> 1 token; stream length UNCHANGED so the
block / instruction runlength columns stay valid).

CRITICAL CONSTRAINT (owner, non-negotiable)
-------------------------------------------
Work on the RAW token-id stream ONLY. This module NEVER routes tokens
through any decoder (FuncTokenList, batch_decode, to_asm_like, ...): all
of those are broken by the same postfix-FP bug. The only codec used is
``base64_to_ndarray_vec`` / ``ndarray_to_base64`` (the pure uint16<->base64
codec). The per-binary vocab (id->string) is parsed DIRECTLY from the CSV's
``vocabulary`` snapshot row, not via any decoder.

The discriminator (asm-tokenizer CONFIRMED EXACT)
-------------------------------------------------
A BROKEN postfix-annotation float = a token whose TYPE (resolved via the
per-binary vocab id->string map) is one of the six floats
{float16, bfloat16, float32, float64, float80, float128} -- NOT
valued_const_v2, NOT value_negative -- whose NEXT token in the stream is
``>= 256`` OR is end-of-stream (no inline-digit token 0..255 follows).
Replace EXACTLY those with the float_annotation id.

SOUNDNESS: a VALUED float always emits ``[type_id, *W payload bytes]`` with
W >= 2 inline-digit tokens (ids < 256, even a 0x00 byte = id 0 which is
< 256) immediately following. Inline digits (0..255) ONLY appear as a
preceding valued type's payload. So float-type id followed by ``< 256`` =
VALUED (never touched); followed by ``>= 256`` or EOS = BROKEN (replaced).

Per-binary id base (legacy vs modern)
--------------------------------------
The first ``vocabulary`` name maps to per-binary id 256 OR 257 depending on
when the CSV was tokenized:

* LEGACY (no ``value_negative``): reserved prefix is 256 digit slots; the
  first name is a real token at id 256. The entire local corpus sampled
  (z3/openssl/curl/nmap/unrar/...) is this era -- names[0] == ``block_v2``.
* MODERN (``value_negative`` pinned at slot 256): reserved prefix is 257;
  the first name is at id 257.

We auto-detect via ``names[0] == "value_negative"`` -- the same rule the
unify loader's ``legacy_no_value_negative`` path uses. Hard-coding 257 (as
an early spec draft did) would mislabel every type id by one on the legacy
corpus, so detection is mandatory. The discriminator's broken/healthy test
is offset-independent; only float-type-id resolution and the new
float_annotation id depend on the correct base.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Pure codec only (NOT a token decoder). base64<->uint16 ids.
from tokenizer.compact_base64_utils import base64_to_ndarray, base64_to_ndarray_vec, ndarray_to_base64

# ---------------------------------------------------------------------------
# Constants (raw-format invariants; no decoder import).
# ---------------------------------------------------------------------------
PREAMBLE = "version=2"
VOCAB_MARKER = "vocabulary"
FUNCTION_HEADER_FIRST = "function_name"
TOKENS_COL = 2  # function-row column index of tokens_base64

# The six v2 float TYPE token canonical strings (token_manager.py basenames).
FLOAT_TYPE_NAMES = frozenset(
    {"float16", "bfloat16", "float32", "float64", "float80", "float128"}
)

FLOAT_ANNOTATION_NAME = "float_annotation"

# Protocol-reserved prefix. Per-binary CSVs are LAZY: the saver strips the
# reserved prefix, so the first ``vocabulary`` name maps to per-binary id
# 256 (legacy, no value_negative) or 257 (modern, value_negative pinned at
# 256). We detect which by inspecting names[0].
RESERVED_DIGIT_COUNT = 256          # ids 0..255 are inline-digit slots
VALUE_NEGATIVE_NAME = "value_negative"

# Inline-digit ceiling: a token id < 256 is an inline-digit payload byte.
INLINE_DIGIT_CEILING = 256

# ``float_annotation`` token-type for the per-binary _id_to_token_type array.
# Authoritative value from asm-tokenizer (origin/main @ cb01cc0):
# ``TokenType.FLOAT_ANNOTATION = 31`` (after VALUE_NEGATIVE=30). It is a
# value-less MODIFIER (family of thread_local=25/vtable=26/code_ptr_table=27),
# arch-agnostic, lazy/first-seen at the next-free per-binary id. unify
# canonicalizes it via the representative-iteration path. Parameterized via
# --annotation-token-type purely so a future enum reshuffle stays a flag, not
# a code edit.
DEFAULT_ANNOTATION_TOKEN_TYPE = 31  # TokenType.FLOAT_ANNOTATION (absolute)

# Platform-instruction-type for the new entry: AGNOSTIC, copied from the
# float TYPE tokens' own pit value in this binary's row[5] rather than a
# hardcoded constant ("copy the floats' pit value to be safe rather than
# reasoning the norm" -- asm-tokenizer). Fallback when the binary has float
# types but (impossibly) no resolvable pit, or for an annotation added to a
# vocab with no float types: AGNOSTIC absolute = -1.
PIT_AGNOSTIC_FALLBACK = -1  # PlatformInstructionTypes.AGNOSTIC (raw int)


# ===========================================================================
# Concern 1 -- per-binary vocab snapshot (id->string + parallel type arrays).
# Owns the ``vocabulary`` row wire format. No decoder.
# ===========================================================================
@dataclass
class VocabSnapshot:
    """A parsed ``vocabulary`` snapshot row of a per-binary CSV.

    Holds only what the fixer needs: the id->name resolution (to find the
    float-type ids) and the parallel type / platform-instruction-type arrays
    (to append a consistent ``float_annotation`` entry). Round-trips
    byte-identically through the codec when unmodified.
    """

    row: list[str]
    legacy_no_value_negative: bool
    start_id: int                    # per-binary id of names[0]
    names: list[str]
    type_norm: int                   # row[2] "norm:" offset
    types: np.ndarray                # absolute _id_to_token_type[start:]
    pit_norm: int                    # row[4] "norm:" offset
    pit: np.ndarray                  # absolute platform-instruction types

    @classmethod
    def parse(cls, row: list[str]) -> "VocabSnapshot":
        assert row and row[0] == VOCAB_MARKER, "not a vocabulary row"
        names = row[1].strip('"').split(",")
        legacy = names[0] != VALUE_NEGATIVE_NAME
        start_id = RESERVED_DIGIT_COUNT if legacy else RESERVED_DIGIT_COUNT + 1
        type_norm = int(row[2].partition("norm:")[2])
        types = base64_to_ndarray(row[3]).astype(np.int64) + type_norm
        pit_norm = int(row[4].partition("norm:")[2])
        pit = base64_to_ndarray(row[5]).astype(np.int64) + pit_norm
        assert len(types) == len(names) == len(pit), (
            f"vocab arrays misaligned: names={len(names)} "
            f"types={len(types)} pit={len(pit)}"
        )
        return cls(
            row=row,
            legacy_no_value_negative=legacy,
            start_id=start_id,
            names=names,
            type_norm=type_norm,
            types=types,
            pit_norm=pit_norm,
            pit=pit,
        )

    @property
    def next_free_id(self) -> int:
        """Per-binary id one past the last registered token."""
        return self.start_id + len(self.names)

    def float_type_ids(self) -> set[int]:
        """Per-binary ids of the six float TYPE tokens present in this vocab."""
        return {
            self.start_id + i
            for i, nm in enumerate(self.names)
            if nm in FLOAT_TYPE_NAMES
        }

    def has_float_annotation(self) -> bool:
        return FLOAT_ANNOTATION_NAME in self.names

    def float_pit_value(self) -> int:
        """The platform-instruction-type the six float TYPE tokens carry in
        this row (they share one AGNOSTIC value). Copied verbatim onto the
        new ``float_annotation`` entry. Falls back to AGNOSTIC absolute when
        the vocab declares no float types."""
        for i, nm in enumerate(self.names):
            if nm in FLOAT_TYPE_NAMES:
                return int(self.pit[i])
        return PIT_AGNOSTIC_FALLBACK

    def with_float_annotation(self, token_type: int) -> tuple[list[str], int]:
        """Return a NEW vocab row (list[str]) with ``float_annotation``
        appended at id ``next_free_id``, plus that assigned id.

        Appends one entry to the name list and one to each parallel array
        (type = ``token_type``, platform-instruction-type copied from the
        binary's float tokens), re-encoding both arrays through the pure
        codec. Idempotent: if the token is already present, returns the row
        unchanged with its id.
        """
        if self.has_float_annotation():
            existing_id = self.start_id + self.names.index(FLOAT_ANNOTATION_NAME)
            return list(self.row), existing_id

        assigned_id = self.next_free_id
        new_names = self.names + [FLOAT_ANNOTATION_NAME]
        new_types = np.concatenate([self.types, [token_type]])
        new_pit = np.concatenate([self.pit, [self.float_pit_value()]])

        new_row = list(self.row)
        new_row[1] = ",".join(new_names)
        new_row[3] = ndarray_to_base64((new_types - self.type_norm).astype(np.uint64))
        new_row[5] = ndarray_to_base64((new_pit - self.pit_norm).astype(np.uint64))
        return new_row, assigned_id


# ===========================================================================
# Concern 2 -- raw token-id stream rewrite (the discriminator). No decoder.
# ===========================================================================
def find_broken_float_positions(
    token_ids: np.ndarray, float_type_ids: set[int]
) -> list[int]:
    """Positions of BROKEN postfix floats in a raw uint16 id stream.

    A position ``p`` is broken iff ``token_ids[p]`` is a float TYPE id AND
    (``p`` is the last token OR ``token_ids[p+1] >= 256``). Pure index logic;
    no decoding, no payload consumption.
    """
    if not float_type_ids:
        return []
    n = len(token_ids)
    out: list[int] = []
    for p in range(n):
        if int(token_ids[p]) in float_type_ids:
            is_eos = p + 1 >= n
            if is_eos or int(token_ids[p + 1]) >= INLINE_DIGIT_CEILING:
                out.append(p)
    return out


def rewrite_stream(
    token_ids: np.ndarray, float_type_ids: set[int], annotation_id: int
) -> tuple[np.ndarray, int]:
    """Return (new_ids, n_replaced). Length-preserving 1->1 swap of every
    broken-float position to ``annotation_id``."""
    positions = find_broken_float_positions(token_ids, float_type_ids)
    if not positions:
        return token_ids, 0
    new_ids = token_ids.astype(np.int64, copy=True)
    for p in positions:
        new_ids[p] = annotation_id
    return new_ids, len(positions)


# ===========================================================================
# Orchestration -- per-CSV fixer.
# ===========================================================================
@dataclass
class FixResult:
    csv_path: Path
    broken_path: Path
    fixed_path: Path
    annotation_id: int
    rows_total: int
    rows_modified: int
    floats_replaced: int
    float_type_ids: set[int]
    noop: bool = False


def _read_rows(path: Path) -> tuple[str, list[str], list[list[str]]]:
    """Return (preamble_line, header_row, body_rows). Body rows include
    both FUNCTION rows and interspersed ``vocabulary`` snapshot rows, in
    file order."""
    with open(path, newline="") as fh:
        preamble = fh.readline()
        assert preamble.strip() == PREAMBLE, (
            f"{path}: expected '{PREAMBLE}' preamble, got {preamble!r}"
        )
        reader = csv.reader(fh)
        header = next(reader)
        assert header[0] == FUNCTION_HEADER_FIRST, (
            f"{path}: unexpected header {header!r}"
        )
        body = [r for r in reader if r]
    return preamble, header, body


def fix_csv(
    csv_path: Path,
    *,
    annotation_token_type: int,
    broken_suffix: str = "_broken",
    dry_run: bool = False,
) -> FixResult:
    """Patch a single ``*_output.csv``.

    The ORIGINAL is renamed to ``*_output<broken_suffix>.csv`` (kept) and the
    fixed content is written to the original path. The float_annotation id is
    derived from the LAST vocab snapshot's ``next_free_id`` and is used to (a)
    rewrite every broken-float position in EVERY function row and (b) register
    the token in the LAST (authoritative) vocab snapshot row, which is the one
    unify reads.
    """
    csv_path = Path(csv_path)
    preamble, header, body = _read_rows(csv_path)

    vocab_indices = [i for i, r in enumerate(body) if r and r[0] == VOCAB_MARKER]
    assert vocab_indices, f"{csv_path}: no vocabulary snapshot row found"

    # Anchor on the LAST vocab snapshot: it is the complete superset and the
    # ONLY row any consumer reads (unify's read_last_line_of_file). The
    # float-type ids and the float_annotation id are both resolved against
    # it, so a single annotation id is valid for the whole file when
    # interpreted (as per-binary CSVs always are) against the final vocab.
    last_vi = vocab_indices[-1]
    last_vocab = VocabSnapshot.parse(body[last_vi])
    float_type_ids = last_vocab.float_type_ids()
    annotation_id = last_vocab.next_free_id

    # --- rewrite function rows --------------------------------------------
    rows_modified = 0
    floats_replaced = 0
    for r in body:
        if r[0] == VOCAB_MARKER:
            continue
        ids = base64_to_ndarray_vec(r[TOKENS_COL]).astype(np.int64)
        new_ids, n_rep = rewrite_stream(ids, float_type_ids, annotation_id)
        if n_rep:
            r[TOKENS_COL] = ndarray_to_base64(new_ids.astype(np.uint64))
            rows_modified += 1
            floats_replaced += n_rep

    broken_path = csv_path.with_name(csv_path.stem + broken_suffix + csv_path.suffix)
    fixed_path = csv_path

    # STRICT NO-OP: a file with no broken floats (clean / inline-only, e.g.
    # curl) is left BYTE-IDENTICAL on disk -- no float_annotation entry, no
    # *_broken.csv copy, no rewrite. float_annotation is registered ONLY when
    # the stream actually uses its id, keeping the vocab edit and the stream
    # edit inseparable.
    if floats_replaced == 0:
        return FixResult(
            csv_path=csv_path, broken_path=broken_path, fixed_path=fixed_path,
            annotation_id=annotation_id, rows_total=len(body) - len(vocab_indices),
            rows_modified=0, floats_replaced=0, float_type_ids=float_type_ids,
            noop=True,
        )

    # --- register float_annotation in the LAST vocab snapshot only --------
    # Intermediate periodic snapshots are historical checkpoints that no
    # consumer reads; registering the global annotation_id there would assign
    # it a DIFFERENT (smaller) per-snapshot id, an inconsistency. The last
    # row is authoritative -- edit only it. with_float_annotation is
    # idempotent, so re-running the fixer is safe.
    new_last_row, registered_id = last_vocab.with_float_annotation(annotation_token_type)
    assert registered_id == annotation_id, (
        f"float_annotation id mismatch: stream used {annotation_id}, "
        f"vocab registered {registered_id}"
    )
    body[last_vi] = new_last_row

    if not dry_run:
        # Keep the original as *_broken.csv (do NOT delete-on-write); write
        # the fixed content to the original path.
        if broken_path.exists():
            raise FileExistsError(
                f"{broken_path} already exists; refusing to overwrite a "
                f"prior original. Resolve manually."
            )
        csv_path.rename(broken_path)
        with open(fixed_path, "w", newline="") as fh:
            fh.write(preamble if preamble.endswith("\n") else preamble + "\n")
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(body)

    return FixResult(
        csv_path=csv_path,
        broken_path=broken_path,
        fixed_path=fixed_path,
        annotation_id=annotation_id,
        rows_total=len(body) - len(vocab_indices),
        rows_modified=rows_modified,
        floats_replaced=floats_replaced,
        float_type_ids=float_type_ids,
        noop=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+", help="*_output.csv file(s)")
    parser.add_argument(
        "--annotation-token-type",
        type=int,
        default=DEFAULT_ANNOTATION_TOKEN_TYPE,
        help=(
            "TokenType enum int for float_annotation in the per-binary "
            "_id_to_token_type array. Default 31 == TokenType.FLOAT_ANNOTATION "
            "(asm-tokenizer origin/main @ cb01cc0). Override only if the enum "
            "value changes."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="scan + report, write nothing"
    )
    args = parser.parse_args(argv)

    csv.field_size_limit(1 << 30)

    for path in args.csv:
        res = fix_csv(
            path,
            annotation_token_type=args.annotation_token_type,
            dry_run=args.dry_run,
        )
        if res.noop:
            status = "NO-OP (no broken floats; file untouched)"
        elif args.dry_run:
            status = "(dry-run)"
        else:
            status = f"-> {res.broken_path.name} kept"
        print(
            f"{res.csv_path}: float_type_ids={sorted(res.float_type_ids)} "
            f"annotation_id={res.annotation_id} "
            f"rows={res.rows_total} rows_modified={res.rows_modified} "
            f"floats_replaced={res.floats_replaced} {status}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
