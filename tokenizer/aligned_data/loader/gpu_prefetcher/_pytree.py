"""Generic pytree tensor-leaf move (no torch, no consumer field names).

Single concern: recurse an OPAQUE batch -- an arbitrary nesting of
dataclass / Mapping / namedtuple / list / tuple holding leaves -- and
apply a move ``fn`` to each LEAF the injected ``is_leaf`` predicate
selects, rebuilding the SAME structure. Every non-leaf, non-container
value passes through unchanged. No field is ever read by name, so the
prefetcher never learns a single consumer batch field name.

``is_leaf`` is injected (the prefetcher defaults it to ``torch.is_tensor``)
so the traversal is fully testable WITHOUT torch: a fake leaf type + a
fake predicate + a tagging ``fn`` exercise the entire walk.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Callable


def _is_namedtuple(obj: Any) -> bool:
    """A tuple subclass carrying ``_fields`` (the namedtuple marker)."""
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def map_tensor_leaves(
    batch: Any,
    fn: Callable[[Any], Any],
    *,
    is_leaf: Callable[[Any], bool],
) -> Any:
    """Recurse an OPAQUE pytree, applying ``fn`` to every leaf.

    Structure is recursed by Python TYPE alone -- dataclass, Mapping,
    namedtuple, and other ``Sequence``s (list / tuple) -- and rebuilt
    with the SAME type. A node for which ``is_leaf`` is True is a leaf:
    ``fn`` is applied to it. Every other non-container leaf (int, str,
    None, ...) passes through UNCHANGED. Dataclass fields are walked via
    :func:`dataclasses.fields`, so a dataclass holding a nested
    dict-of-tensors AND a plain int scalar is handled with ZERO special
    casing.
    """
    if is_leaf(batch):
        return fn(batch)

    # Dataclass INSTANCE (not the class object): rebuild via its fields.
    if dataclasses.is_dataclass(batch) and not isinstance(batch, type):
        moved = {
            f.name: map_tensor_leaves(getattr(batch, f.name), fn, is_leaf=is_leaf)
            for f in dataclasses.fields(batch)
        }
        return dataclasses.replace(batch, **moved)

    # Mapping: rebuild same mapping type, keys untouched, values recursed.
    if isinstance(batch, Mapping):
        items = {
            k: map_tensor_leaves(v, fn, is_leaf=is_leaf) for k, v in batch.items()
        }
        try:
            return type(batch)(items)
        except TypeError:
            # Mappings not constructible from a dict (rare) -> plain dict.
            return items

    # namedtuple: rebuild positionally via the subclass constructor.
    if _is_namedtuple(batch):
        return type(batch)(
            *(map_tensor_leaves(v, fn, is_leaf=is_leaf) for v in batch)
        )

    # Other sequences (list / tuple), but NOT str/bytes (those are leaves).
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
        mapped = [map_tensor_leaves(v, fn, is_leaf=is_leaf) for v in batch]
        try:
            return type(batch)(mapped)
        except TypeError:
            return type(batch)(*mapped) if isinstance(batch, tuple) else list(mapped)

    # Plain non-leaf, non-container value: pass through.
    return batch
