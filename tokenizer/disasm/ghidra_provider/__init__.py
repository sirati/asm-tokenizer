"""Ghidra-based disassembly provider using pyghidra (headless, no IPC).

Re-exports the public surface that the legacy ``ghidra_provider.py``
module exposed. Internal submodule layout (single-concern split,
task #63):

- ``section_classify``: section-name / string-DataType / vtable helpers.
- ``metadata_view``: ``_GhidraAddressMetadataView`` storage + read-only
  typed properties.
- ``metadata_lookup``: ``GhidraMetadataLookup`` classifier +
  slot-target + switch-table cache.
- ``mnemonic``: ``_RegisterMap`` + mnemonic split / alias /
  x86-prefix-byte helpers.
- ``mem_decompose``: per-ISA memory-operand decomposition helpers.
- ``prefix_build``: ISA detection + FP-type computation + per-ISA
  typed-prefix-list builders.
- ``decode_helper``: ``_GhidraDecodeHelper`` injected into view wrappers.
- ``provider``: ``GhidraDisassemblyProvider`` entrypoint.
"""

from tokenizer.disasm.ghidra_provider.metadata_lookup import GhidraMetadataLookup
from tokenizer.disasm.ghidra_provider.metadata_view import _GhidraAddressMetadataView
from tokenizer.disasm.ghidra_provider.provider import GhidraDisassemblyProvider

__all__ = [
    "GhidraDisassemblyProvider",
    "GhidraMetadataLookup",
    "_GhidraAddressMetadataView",
]
