"""BinarySession integration: ``line_to_provider`` metadata key.

Verifies the wiring that loads the per-binary
``<binary>_extern_providers.txt`` sidecar into the session metadata
bag at ``BinaryDataset.open_session()`` time, so consumers
(e.g. the inspector's ``InlineCallEntry.provider`` resolution) read
the library mapping via ``session.get_metadata("line_to_provider")``.
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset

from ._session_fixture import synthetic_binary  # noqa: F401


def test_session_exposes_empty_extern_providers_via_metadata(
    synthetic_binary,
) -> None:
    """Corpus with no extern calls -> empty ``line_to_provider`` mapping."""
    fb = synthetic_binary
    ds = BinaryDataset(fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"])
    with ds.open_session() as sess:
        providers = sess.get_metadata("line_to_provider")
    assert providers == {}


def test_session_exposes_populated_extern_providers_via_metadata(
    synthetic_binary,
) -> None:
    """Rewrite the sidecar with two libraries; session metadata reflects them.

    The corpus builder emits an empty sidecar when no specs declare
    extern calls. Re-stamping the sidecar file in-place after corpus
    build exercises the full loader path (prelude validation + 1-indexed
    materialization) without needing extern-emitting specs.
    """
    fb = synthetic_binary
    base_path: Path = fb["base_path"]
    binary_name: str = fb["binary_name"]

    reg = ExternProviderRegistry()
    reg.add("libc.so.6")
    reg.add("libm.so.6")
    reg.write_sidecar(base_path, binary_name)

    ds = BinaryDataset(base_path, binary_name, vocab_manager=fb["vocab"])
    with ds.open_session() as sess:
        providers = sess.get_metadata("line_to_provider")

    assert providers == {1: "libc.so.6", 2: "libm.so.6"}
