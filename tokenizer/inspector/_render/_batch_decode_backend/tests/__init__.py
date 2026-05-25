"""Tests for the BatchDecodeBackend subpackage.

Plan reference: ``inspector-render-backends.md`` §10 -- the per-
subpackage tests subdir mandated for backend internals. The loader
layer's ``test_fid_per_category_counts.py`` + ``test_auto_size.py``
already pin the sidecars these tests consume; this subdir pins the
backend-side invariants:

* :mod:`test_fid_table` -- ``FidBaseTable`` cumsum + per-row base +
  recursive-call dedup contract.
* :mod:`test_number_hex_format` -- ``chunks_to_hex_bits`` cross-
  backend hex parity vs ``Inner.to_asm_like``.
* :mod:`test_row_walk` -- end-to-end row walk on synthetic
  ``BatchDecodeResult`` fixtures (pre-allocated entry block, multi-
  chunk trailing-slot placeholder, n_axis BLOCK_V2 guard, padding).
* :mod:`test_backend` -- ``BatchDecodeBackend`` Protocol compliance
  + lifecycle (closed-flag, lazy decode).
"""
