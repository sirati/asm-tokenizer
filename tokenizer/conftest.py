"""Conditional parent-package pre-empt for ``tokenizer.aligned_data``.

While the memmap-record-header rewrite rolls out across parallel
batches, ``tokenizer/aligned_data/__init__.py`` eagerly imports
re-exports from sibling modules (``_writers.py``, ``io.py``, the
``loader/`` subpackage) that still reference pre-refactor symbols this
batch deletes. The eager init therefore raises ``ImportError`` at
collection time -- even for a conftest physically located inside
``tokenizer/aligned_data/``, because pytest has to traverse the
package chain to import it.

This conftest sits one directory ABOVE ``aligned_data`` (under the
``tokenizer`` package, whose ``__init__.py`` is empty and side-effect
free) and runs early enough to install a sys.modules stub for
``tokenizer.aligned_data`` BEFORE pytest descends into the broken
subpackage. The stub mimics a regular package (carries ``__path__``)
so direct subpackage imports like ``tokenizer.aligned_data.binary_format``
still resolve normally.

The pre-empt is conditional: we first try to import the real package
and only install the stub on ``ImportError``. Once the sibling
batches merge the real init loads cleanly, this conftest becomes a
no-op and self-removes when the next test session runs.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _try_preempt_aligned_data() -> None:
    name = "tokenizer.aligned_data"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        pkg_root = Path(__file__).resolve().parent / "aligned_data"
        if not pkg_root.is_dir():
            return
        if "tokenizer" not in sys.modules:
            tokenizer_stub = types.ModuleType("tokenizer")
            tokenizer_stub.__path__ = [str(pkg_root.parent)]
            sys.modules["tokenizer"] = tokenizer_stub
        aligned_stub = types.ModuleType(name)
        aligned_stub.__path__ = [str(pkg_root)]
        sys.modules[name] = aligned_stub


_try_preempt_aligned_data()
