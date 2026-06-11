"""Count the serialized vocabulary entries in a per-binary output CSV.

Single concern: given a ``<fullname>_output.csv`` path, return how many
vocabulary tokens are written on the wire.  No DB, no axis parsing.

A per-binary tokenize output serialises its :class:`VocabularyManager`
as the **last** CSV row of the file (the "vocab-def" row).  The wire
vocabulary cell (``row[1]``, comma-separated) is the canonical token
list with the protocol-reserved prefix stripped: the saver omits the
256 inline-digit slots PLUS ``value_negative`` at slot 256
(``VocabularyManager._V2_RESERVED_TOKEN_COUNT`` = 257 slots), and the
loader reconstitutes them.  The count returned here is therefore the
**serialized wire vocabulary count, excluding those 257 reserved
non-serialized slots** — equivalently ``len(VocabularyManager.id_to_token)
- 257`` for a round-tripped manager (verified equal).

The row is located and validated with the tokenizer's own loader
primitives (:func:`read_last_line_of_file` to read just the tail via
memmap, :func:`is_vocab_def` to confirm the row is a real vocab-def);
this module adds only the count and never re-implements the wire format.
The per-binary platform needed by :func:`is_vocab_def` is derived from
the filename prefix using the same arch catalog the loader uses, so a
malformed/aliased prefix degrades to ``None`` rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.arch import Platform
from tokenizer.arch_translation import all_known_arch_strings, arch_to_platform
from tokenizer.vocab_unifier.loader import is_vocab_def, read_last_line_of_file

# Longest-first so a longer arch alias is matched before a shorter one
# that prefixes it (mirrors the loader's own detection ordering).
_KNOWN_ISAS: tuple[str, ...] = tuple(
    sorted(all_known_arch_strings(), key=len, reverse=True)
)

# Index of the wire vocabulary cell in a vocab-def row (``save_vocabulary``
# writes "vocabulary" at 0 and the comma-joined token list at 1).
_VOCAB_CELL_INDEX = 1


def _platform_from_filename(filename: str) -> Platform | None:
    """Detect the per-binary :class:`Platform` from a fullname prefix,
    reusing the loader's arch catalog.  ``None`` when no known arch
    prefixes the name (the file then yields a NULL vocab_size)."""
    for isa in _KNOWN_ISAS:
        if filename.startswith(isa + "-"):
            return arch_to_platform(isa)
    return None


def count_vocab(csv_path: Path) -> int | None:
    """Return the serialized wire vocabulary entry count for a per-binary
    ``_output.csv``, or ``None`` when the file has no readable/valid
    vocab-def row (absent vocab ⇒ NULL, never 0)."""
    platform = _platform_from_filename(csv_path.name)
    if platform is None:
        return None
    try:
        last_line = read_last_line_of_file(csv_path)
    except Exception:
        return None
    valid, row = is_vocab_def(last_line, platform)
    if not valid:
        return None
    # The saver builds this cell with ``",".join(id_to_token[start:])`` and
    # token names never contain commas, so a plain split is the exact
    # inverse.  An empty cell ⇒ 0 serialized tokens (not 1).
    cell = row[_VOCAB_CELL_INDEX]
    if cell == "":
        return 0
    return len(cell.split(","))
