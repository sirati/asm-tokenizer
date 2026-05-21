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
    # For each real-token position p (in order), returns the inline-data run
    # length starting at p+1. The final real-token's trailing run is the
    # caller's concern (no position p+1 exists when p == N-1, so the slice
    # implicitly drops it via real_mask[:-1]).
    return runlen[1:][real_mask[:-1]]
