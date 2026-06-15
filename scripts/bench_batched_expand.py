"""Micro-benchmark: old per-node expand vs new batched expand.

Builds a ~300-node flat body stream of realistic v2 carriers + payloads,
then times (a) the OLD per-node drive (build_inline_decode_state +
expand_tokens in a Python loop) against (b) the NEW batched_expand over
the whole flat stream. Reports the expand-loop speedup. Pure perf probe;
correctness is the test suite's job.
"""

from __future__ import annotations

import time

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    _CALL_TARGET_TYPE_TO_CATEGORY,
)
from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    expand_tokens,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    build_inline_decode_state,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._batched_expand import (
    batched_expand,
)
from tokenizer.aligned_data.loader.vector_batch._scatter._expand import (
    _SELF_TOKEN_LUT,
)
from tokenizer.token_manager import VocabularyManager


_VC2 = VocabularyManager._V2_NUMBER_BLOCK_START
_F128 = _VC2 + VocabularyManager._V2_NUMBER_BLOCK_COUNT - 1
_SIGN = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID
_IDENT = VocabularyManager._V2_IDENTITY_BLOCK_START
_INSN = VocabularyManager._V2_EAGER_BLOCK_END + 3  # a non-carrier real id


class _Shim:
    def __init__(self, state, category):
        self.state = state
        self.encounter_category = category


def _random_body(rng):
    """A plausible v2 body: opens with a real carrier, then a mix of
    instruction reps, identity carriers, and VC2 / F128 number sources
    with inline-digit payloads."""
    out = [int(_INSN)]
    for _ in range(rng.integers(6, 20)):
        r = rng.random()
        if r < 0.3:
            out.append(int(_INSN + rng.integers(0, 40)))
        elif r < 0.5:
            out.append(int(_IDENT + rng.integers(0, 6)))
        elif r < 0.75:
            out.append(int(_VC2))
            out.extend(int(d) for d in rng.integers(0, 256, rng.integers(1, 12)))
            if rng.random() < 0.3:
                out.append(int(_SIGN))
        else:
            out.append(int(_F128))
            hi = 0x3FFF if rng.random() < 0.7 else 0x7FFF
            out.append((hi >> 8) & 0xFF)
            out.append(hi & 0xFF)
            out.extend([0] * 14)
    return out


def _build(n_nodes, seed=0):
    rng = np.random.default_rng(seed)
    bodies = [_random_body(rng) for _ in range(n_nodes)]
    counts = [len(b) for b in bodies]
    rec = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=rec[1:])
    raw = np.concatenate(
        [np.asarray(b, dtype=np.uint16) for b in bodies]
    ).astype(np.uint16)
    edge_types = rng.integers(0, 2, n_nodes).astype(np.uint8)  # LOCAL/PLT
    return raw, rec, edge_types


def _old_path(raw, rec, edge_types):
    n = rec.size - 1
    pieces = []
    for i in range(n):
        body = raw[rec[i] : rec[i + 1]]
        state = build_inline_decode_state(body, format_version=1)
        cat = _CALL_TARGET_TYPE_TO_CATEGORY[CallTargetType(int(edge_types[i]))]
        res = expand_tokens(_Shim(state, cat))
        pieces.append(res.expanded_token_ids)
    return np.concatenate(pieces)


def _new_path(raw, rec, edge_types):
    self_ids = _SELF_TOKEN_LUT[edge_types].astype(np.uint16)
    return batched_expand(raw, rec, self_ids).expanded


def _time(fn, *args, reps):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    n_nodes = 300
    raw, rec, edge_types = _build(n_nodes)
    # Sanity: identical output.
    assert np.array_equal(
        _old_path(raw, rec, edge_types), _new_path(raw, rec, edge_types)
    ), "old/new expand diverge"

    for fn in (_old_path, _new_path):  # warm imports/JIT
        fn(raw, rec, edge_types)

    reps = 200
    t_old = _time(_old_path, raw, rec, edge_types, reps=reps)
    t_new = _time(_new_path, raw, rec, edge_types, reps=reps)
    print(f"nodes={n_nodes}  raw_tokens={raw.size}")
    print(f"old per-node expand : {t_old * 1e3:.3f} ms")
    print(f"new batched expand  : {t_new * 1e3:.3f} ms")
    print(f"speedup             : {t_old / t_new:.2f}x")


if __name__ == "__main__":
    main()
