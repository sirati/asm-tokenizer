"""Phase-0c stub smoke tests.

Two invariants:

1. Every stub module imports cleanly (catches syntax errors + import-time
   regressions when Phase-0b's ``_types`` is or isn't present).
2. Every public stage function raises :class:`NotImplementedError` on the
   minimal call that reaches its body. This catches the common stub
   regression of "someone accidentally wrote a real body instead of a
   ``raise``".

The arguments passed below are intentionally dummies — the goal is to reach
the function body, not to satisfy its semantic preconditions. Phase 1..4
subagents replace these tests with real stage tests as they fill in each
body.
"""

from __future__ import annotations

import importlib

import pytest


STUB_MODULES = (
    "tokenizer.aligned_data.loader.batch_decode._section_walk",
    "tokenizer.aligned_data.loader.batch_decode._length_predict",
    "tokenizer.aligned_data.loader.batch_decode._bulk_bytes",
    "tokenizer.aligned_data.loader.batch_decode._assemble",
    "tokenizer.aligned_data.loader.batch_decode._entry",
    "tokenizer.aligned_data.loader.batch_decode",
)


@pytest.mark.parametrize("modname", STUB_MODULES)
def test_stub_module_imports_cleanly(modname: str) -> None:
    """Module must import without raising — no syntax errors, no import-time
    side effects that fail in isolation."""

    mod = importlib.import_module(modname)
    assert mod is not None


def test_assemble_batch_raises_not_implemented() -> None:
    from tokenizer.aligned_data.loader.batch_decode._assemble import assemble_batch

    with pytest.raises(NotImplementedError):
        assemble_batch(stage3=None, context_len=0)  # type: ignore[arg-type]


def test_batch_decode_entry_raises_not_implemented() -> None:
    from tokenizer.aligned_data.loader.batch_decode._entry import batch_decode

    with pytest.raises(NotImplementedError):
        batch_decode(
            session=None,  # type: ignore[arg-type]
            section_pointers=[],
            num_variants_per_section=1,
            context_len=0,
            max_depth=0,
        )
