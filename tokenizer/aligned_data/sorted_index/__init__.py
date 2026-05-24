"""Per-binary sorted-length index for length-bucketed dataloader sampling.

Public API (grown incrementally as the package fills in):

* :func:`encode_sorted_index` -- pure numpy encode of a u32 length array
  into the on-wire blob.
* :func:`parse_header` -- pure numpy decode of the wire header.

Both names are pure wire-format concerns and live in :mod:`_wire`.
"""

from __future__ import annotations

from ._wire import encode_sorted_index, parse_header

__all__ = ["encode_sorted_index", "parse_header"]
