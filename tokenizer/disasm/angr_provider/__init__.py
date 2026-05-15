"""angr-based disassembly provider.

Re-exports the public surface that the legacy ``angr_provider.py`` module
exposed. Internal submodule layout (single-concern split, task #63):

- ``metadata_view``: ``_AngrAddressMetadataView`` storage + typed properties.
- ``op_classify``: archinfo / Capstone op-type / ARM-shift classifiers
  + import-time ``fp_type = None`` Capstone stamping.
- ``prefixes``: per-platform prefix builders + concrete prefix subclasses.
- ``views``: owned function / block / instruction / operand cursors +
  container views.
- ``provider``: ``AngrDisassemblyProvider`` entrypoint.

``AngrMetadataLookup`` / ``AddressMetaDataLookup`` are resolved lazily via
module-level ``__getattr__`` to break the ``angr_provider`` <->
``address_meta_data_lookup`` import cycle.
"""

from tokenizer.disasm.angr_provider.metadata_view import _AngrAddressMetadataView
from tokenizer.disasm.angr_provider.provider import (
    AngrDisassemblyProvider,
    _import_lookup_classes,
)


def __getattr__(name: str):
    """Module-level ``__getattr__`` for re-export laziness.

    ``from tokenizer.disasm.angr_provider import AngrMetadataLookup``
    triggers this on first access; the import resolves at that point so
    we don't pay the ``address_meta_data_lookup`` cost at module load.
    """
    if name in {"AngrMetadataLookup", "AddressMetaDataLookup"}:
        AngrMetadataLookup, AddressMetaDataLookup = _import_lookup_classes()
        return {"AngrMetadataLookup": AngrMetadataLookup, "AddressMetaDataLookup": AddressMetaDataLookup}[name]
    raise AttributeError(f"module 'tokenizer.disasm.angr_provider' has no attribute {name!r}")


__all__ = [  # noqa: F822 - "AngrMetadataLookup" / "AddressMetaDataLookup" resolved by __getattr__
    "AddressMetaDataLookup",
    "AngrDisassemblyProvider",
    "AngrMetadataLookup",
    "_AngrAddressMetadataView",
]
