"""Variant-axis tokens — discovery, encoding, and bin-record I/O.

Per plan ``memoized-booping-wren.md``, this package owns four
concerns split across four submodules:

* ``prefixes``  — prefix grammar + ``build_axis_strings``
* ``inventory`` — discovery-time distinct-token accumulator
* ``encoder``   — pure encode/decode against a ``VocabularyManager``
* ``record``    — handle-level read/write helpers

The public surface re-exported here is the set Batches 3 (unifier)
and 5 (memmap-builder + dataloader) need. Internal helpers stay
unexported so the boundary is auditable.
"""

from .encoder import decode_record, encode_record
from .inventory import VariantInventory
from .prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    N_POSITIONAL_AXES,
    OPT_PREFIX,
    build_axis_strings,
    build_metadata_tokens,
)
from .record import read_record, write_record

__all__ = [
    "ARCH_PREFIX",
    "COMP_PREFIX",
    "CVER_PREFIX",
    "N_POSITIONAL_AXES",
    "OPT_PREFIX",
    "VariantInventory",
    "build_axis_strings",
    "build_metadata_tokens",
    "decode_record",
    "encode_record",
    "read_record",
    "write_record",
]
