"""Single source of truth for the memmap output-chain format version.

This version covers every artifact produced by the memmap-output chain:
the unified vocab CSV, per-binary sections CSV preludes, the slim variants
CSV prelude, and the ``_index.bin`` prelude. Bumping the constant here is
intended to cascade: writers import it for the version they stamp, readers
import it for the version they assert, and a bump is a one-line edit plus a
full cascade-rebuild migration of any persisted artifacts.
"""

MEMMAP_FORMAT_VERSION: int = 1
