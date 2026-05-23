"""Single-function decode pass: raw u16 token stream -> ``DecodedFunction``.

This package was carved out of the original ``extract.py`` module to
honour the project's per-file size cap.  Each submodule owns one
concern (staging dataclass, shared occurrence iterator, identity arm,
number arm, orchestrator); the public surface stays unchanged and is
re-exported here so callers keep using
``tokenizer.aligned_data.loader.decoded.extract``.

See :mod:`._orchestrator` for the algorithm-level module docstring.
"""

from ._orchestrator import _decode_to_staging, decode_raw_tokens
from ._staging import _StagingDecoded

__all__ = ["decode_raw_tokens"]
