"""Out-of-band decoded view of one function (post-splice or single-function).

Pure value-carrier. Construction is owned by ``extract.py`` (single function,
no splicing) and ``splice.py`` (multi-function, depth-capped). Consumers
MUST treat every ndarray field as read-only -- the arrays are referenced,
not copied, by both producers and the splicer's identity-rebase + multi-chunk
number-alignment arithmetic; in-place mutation by a consumer would corrupt
that bookkeeping for any subsequent splice pass that re-uses the same view.

Invariants live in ``__post_init__`` (see plan ## Locked-in decisions items
1, 3, 7 + ## Module layout row for this file).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from tokenizer.tokens import Category


@dataclass(frozen=True)
class DecodedFunction:
    """Frozen out-of-band decoded view of a (possibly spliced) function body.

    Fields:
        real_tokens: ``uint16[Nr]`` -- the strip-and-promote output stream.
            Only positions whose original raw-token id was >= 256 (real
            tokens) plus the in-place-promoted slots for multi-chunk
            numbers (see plan Locked-in decision 2). Inline-digit bytes
            are absent.
        identities: ``dict[Category, uint16[Ni_c]]`` -- one array per
            identity-owning category. Length matches the count of that
            category's real-token occurrences in ``real_tokens``. Sentinel
            ``0xFFFF`` flags an unresolved / overflowing identity (see
            plan Locked-in decision 7).

            Post-splice semantics (plan Decisions 22 + 28 + 29):

            * For categories in ``FID_KEYED_CATEGORIES``
              (``LOCAL_FUNC``, ``PLT_FUNC``, ``EXT_FUNC``) the pre-
              compaction value at each slot is the callee's function
              identity (FID) -- equal to the matching
              ``Section.call_targets[*].function_name_ptr``. Same FID
              everywhere a given callee is referenced, regardless of
              where in the splice tree the reference originates.
            * For the other five categories (``BLOCK``,
              ``RO_DATA_PTR``, ``RW_DATA_PTR``, ``STRING_PTR``,
              ``JUMP_TABLE``) the pre-compaction value is the
              per-function encoder counter offset by a running max
              across the splice tree (legacy rebase path).

            All eight categories then run through the splicer's per-
            Category compaction pass, which maps the value space to a
            dense ``[0, K)`` range (sentinel-preserving) so the final
            arrays exposed here are uniformly ``uint16`` regardless of
            the upstream identity domain.
        numbers_significant: ``uint64[Nn]`` -- one entry per number-token
            occurrence in ``real_tokens`` (after multi-chunk promotion).
        numbers_sign_exponent: ``uint32[Nn]`` -- paired with
            ``numbers_significant``; identical length.
        func_name: human-readable name of the root function (callees'
            names are dropped when this view is the result of a splice).
        metadata: free-form root-function metadata dict.

    Consumers MUST treat the arrays as read-only; mutation breaks Splice's
    identity-rebase arithmetic and the multi-chunk number alignment.
    """

    real_tokens: np.ndarray
    identities: Dict[Category, np.ndarray]
    numbers_significant: np.ndarray
    numbers_sign_exponent: np.ndarray
    func_name: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        # --- Paired number-array shape (plan Locked-in decision 3) ---
        n_sig = len(self.numbers_significant)
        n_exp = len(self.numbers_sign_exponent)
        if n_sig != n_exp:
            raise ValueError(
                "numbers_significant and numbers_sign_exponent must be the "
                f"same length; got {n_sig} vs {n_exp}"
            )

        # --- Strict dtypes on the three numpy fields ---
        if self.real_tokens.dtype != np.uint16:
            raise ValueError(
                f"real_tokens must be dtype uint16; got {self.real_tokens.dtype}"
            )
        if self.numbers_significant.dtype != np.uint64:
            raise ValueError(
                "numbers_significant must be dtype uint64; got "
                f"{self.numbers_significant.dtype}"
            )
        if self.numbers_sign_exponent.dtype != np.uint32:
            raise ValueError(
                "numbers_sign_exponent must be dtype uint32; got "
                f"{self.numbers_sign_exponent.dtype}"
            )

        # --- Identities: exactly the 8 Category members, each uint16 ---
        expected = set(Category)
        actual_keys = set(self.identities.keys())

        non_category = {k for k in actual_keys if not isinstance(k, Category)}
        if non_category:
            raise ValueError(
                "identities contains keys that are not Category members: "
                f"{sorted(repr(k) for k in non_category)}"
            )

        missing = expected - actual_keys
        if missing:
            missing_names = sorted(c.name for c in missing)
            raise ValueError(
                "identities is missing required Category keys: "
                f"{missing_names}"
            )

        extra = actual_keys - expected
        if extra:
            # Reachable only via Category-typed keys that somehow aren't
            # canonical members (e.g. a hand-rolled duplicate enum). The
            # non_category check above already covers non-Category objects.
            raise ValueError(
                f"identities contains unexpected Category keys: {sorted(extra)}"
            )

        for cat, arr in self.identities.items():
            if arr.dtype != np.uint16:
                raise ValueError(
                    f"identities[{cat.name}] must be dtype uint16; got {arr.dtype}"
                )
            # uint16 naturally constrains values to [0, 0xFFFF]; assert to
            # make the sentinel contract unambiguous on the consumer side.
            if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 0xFFFF):
                raise ValueError(
                    f"identities[{cat.name}] values must lie in [0, 0xFFFF]"
                )

        # --- func_name + metadata sanity ---
        if not isinstance(self.func_name, str) or not self.func_name:
            raise ValueError("func_name must be a non-empty str")
        if not isinstance(self.metadata, dict):
            raise ValueError(
                f"metadata must be a dict; got {type(self.metadata).__name__}"
            )
