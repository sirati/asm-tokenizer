import numpy as np


def run_lengths(mask):
    """### restriction:
    1. first position in input is always False
    2. max run length must fit into u16
    """
    assert not mask[..., 0].max()
    if mask.shape[-1] == 1:
        return mask.astype(np.uint16)

    flip = mask[..., :-1] != mask[..., 1:]
    up_flip = flip & mask[..., 1:]
    down_flip = flip
    del flip  # rename the reused flip buffer for clarity
    # important we reuse fill down_flip left to right, so we do not override values we still need
    down_flip[..., :-1] = down_flip[..., 1:] & mask[..., 1:-1]
    down_flip[..., -1] = mask[..., -1]  # final True implicitly succeeded by False

    up_idx = np.nonzero(up_flip)
    down_idx = np.nonzero(down_flip)
    out = np.zeros_like(mask, dtype=np.uint16)
    out[..., 1:][up_idx] = down_idx[-1] - up_idx[-1] + 1
    return out


def inline_data_runlength_after_real_tokens(runlen, real_mask):
    """For each real-token position p in input order, return the inline-data
    run length starting at p+1.

    Output shape is ``(real_mask.sum(),)`` — one entry per real token in
    order.  Real tokens at non-final positions get the run length at p+1
    from ``runlen``.  A real token at the FINAL position has no p+1 slot,
    so its entry is 0 (zero-padded by construction).  This makes the
    helper self-contained: callers do not need to pad the tail.
    """
    out = np.zeros(int(real_mask.sum()), dtype=np.uint16)
    # Real tokens at positions strictly less than N-1 (i.e. those with a
    # valid p+1 slot) populate the leading entries of ``out``.  When the
    # FINAL position is a real token, its slot in ``out`` is the trailing
    # zero-pad — no write needed.
    non_last_real_count = int(real_mask[:-1].sum())
    out[:non_last_real_count] = runlen[1:][real_mask[:-1]]
    return out
