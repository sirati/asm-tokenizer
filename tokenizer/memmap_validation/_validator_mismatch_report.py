"""Token-mismatch report formatter.

Single concern: turn a ``(memmap_tokens, csv_tokens)`` disagreement
into a human-readable diff block the validator slots into its
``stats.errors`` list. Lives in its own module so the orchestrator
(``validator.py``) stays focused on the per-binary control flow and
the ASM-decoding side-quest (``FunctionTokenList``) does not bleed
into the orchestrator's import surface.

The formatter is content-agnostic about WHICH function the tokens
belong to; callers prepend a "Tokens mismatch for {func_name} version
{vkey}" header line before appending the formatted block to
``stats.errors``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..function_token_list import FunctionTokenList
from ..token_manager import VocabularyManager


def format_token_mismatch(
    memmap_tokens: np.ndarray,
    csv_tokens: np.ndarray,
    vocab_manager: Optional[VocabularyManager] = None,
) -> str:
    """Format a detailed token mismatch error message.

    Shows where tokens start to differ with context, using ASM-like format if vocab available.
    """
    min_len = min(len(memmap_tokens), len(csv_tokens))
    max_display = min(min_len, 2000)

    # Find first mismatch position
    mismatch_pos = None
    for i in range(min_len):
        if memmap_tokens[i] != csv_tokens[i]:
            mismatch_pos = i
            break

    if mismatch_pos is None:
        # Lengths differ but all common tokens match
        return (
            f"Token count mismatch: memmap has {len(memmap_tokens)}, CSV has {len(csv_tokens)}\n"
            f"  All first {min_len} tokens match correctly"
        )

    # Show context: 10 tokens before mismatch, then the mismatched section (up to 100 more tokens)
    context_start = max(0, mismatch_pos - 10)
    context_end = min(max_display, mismatch_pos + 100)

    result = [
        f"Token mismatch at position {mismatch_pos} (total lengths: memmap={len(memmap_tokens)}, csv={len(csv_tokens)})"
    ]

    if vocab_manager is not None:
        try:
            # Use FunctionTokenList to properly format tokens
            # Create dummy runlengths for display purposes
            dummy_block_rl = np.array([len(memmap_tokens)], dtype=np.uint16)
            dummy_insn_rl = np.ones(len(memmap_tokens), dtype=np.uint8)

            # Show matching prefix if there is one
            if mismatch_pos > 0:
                prefix_tokens = memmap_tokens[context_start:mismatch_pos]
                try:
                    prefix_func = FunctionTokenList.reconstruct_func_from_raw_bytes(
                        prefix_tokens,
                        np.array([len(prefix_tokens)], dtype=np.uint16),
                        np.ones(len(prefix_tokens), dtype=np.uint8),
                        vocab_manager,
                    )
                    prefix_str = " ".join(token.to_asm_like() for token in prefix_func.iter_tokens())
                    result.append(f"  Matching prefix [{context_start}:{mismatch_pos}]: {prefix_str}")
                except Exception:
                    result.append(f"  Matching prefix [{context_start}:{mismatch_pos}]: {list(prefix_tokens)}")

            # Show mismatched section from memmap
            memmap_section = memmap_tokens[mismatch_pos:context_end]
            try:
                memmap_func = FunctionTokenList.reconstruct_func_from_raw_bytes(
                    memmap_section,
                    np.array([len(memmap_section)], dtype=np.uint16),
                    np.ones(len(memmap_section), dtype=np.uint8),
                    vocab_manager,
                )
                memmap_str = " ".join(token.to_asm_like() for token in memmap_func.iter_tokens())
                result.append(f"  Memmap [{mismatch_pos}:{context_end}]: {memmap_str}")
            except Exception:
                result.append(f"  Memmap [{mismatch_pos}:{context_end}]: {list(memmap_section)}")

            # Show mismatched section from CSV
            csv_section = csv_tokens[mismatch_pos:context_end]
            try:
                csv_func = FunctionTokenList.reconstruct_func_from_raw_bytes(
                    csv_section,
                    np.array([len(csv_section)], dtype=np.uint16),
                    np.ones(len(csv_section), dtype=np.uint8),
                    vocab_manager,
                )
                csv_str = " ".join(token.to_asm_like() for token in csv_func.iter_tokens())
                result.append(f"  CSV    [{mismatch_pos}:{context_end}]: {csv_str}")
            except Exception:
                result.append(f"  CSV    [{mismatch_pos}:{context_end}]: {list(csv_section)}")

        except Exception as e:
            # Fallback to raw token IDs if vocab reconstruction fails
            result.append(f"  (Failed to use vocab: {e})")
            result.append(f"  Memmap [{mismatch_pos}:{context_end}]: {list(memmap_tokens[mismatch_pos:context_end])}")
            result.append(f"  CSV    [{mismatch_pos}:{context_end}]: {list(csv_tokens[mismatch_pos:context_end])}")
    else:
        # No vocab manager - show raw token IDs
        if mismatch_pos > 0:
            prefix_tokens = memmap_tokens[context_start:mismatch_pos]
            result.append(f"  Matching prefix [{context_start}:{mismatch_pos}]: {list(prefix_tokens)}")

        memmap_section = memmap_tokens[mismatch_pos:context_end]
        csv_section = csv_tokens[mismatch_pos:context_end]

        result.append(f"  Memmap [{mismatch_pos}:{context_end}]: {list(memmap_section)}")
        result.append(f"  CSV    [{mismatch_pos}:{context_end}]: {list(csv_section)}")

    return "\n".join(result)
