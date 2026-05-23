"""Mid-pipeline staging dataclass for the decode pass.

Single concern of this module: define the :class:`_StagingDecoded` view
that the identity-arm + number-arm decode produces before per-Category
compaction lowers it to a :class:`DecodedFunction`.  Splitting this out
keeps the orchestrator + per-arm modules free of dataclass boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from tokenizer.tokens import Category


# u32 sentinel for FID-keyed identity staging.  Compaction (in splice.py)
# folds these positions back to the public uint16 sentinel ``0xFFFF`` in
# the final :class:`DecodedFunction`.
_IDENTITY_SENTINEL_U32 = np.uint32(0xFFFFFFFF)


@dataclass(frozen=True)
class _StagingDecoded:
    """Private mid-pipeline view: same fields as :class:`DecodedFunction`
    but with NO dtype invariant on the per-Category identity arrays.

    The splice walker concatenates these verbatim and runs per-Category
    compaction at the top level; the public :class:`DecodedFunction`
    is constructed only after compaction so the u16 invariant holds for
    consumers.  Treat the arrays as read-only -- consumers MUST NOT
    mutate them (the splicer reuses the same arrays across the concat
    step).
    """

    real_tokens: np.ndarray
    identities: Dict[Category, np.ndarray]
    numbers_significant: np.ndarray
    numbers_sign_exponent: np.ndarray
    func_name: str
    metadata: Dict[str, Any]
