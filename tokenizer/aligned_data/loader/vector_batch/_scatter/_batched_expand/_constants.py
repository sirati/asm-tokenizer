"""Unified-vocab layout constants for the batched expand machinery.

Single concern: the SAME source-of-truth the scalar kernels resolve (see
``_expand_tokens`` / ``_inline_decode_state`` / ``_bulk_expand_lengths``);
a canonical-layout shift surfaces here as a constant-import update + a
test cascade. Both the state-field math (:mod:`._state_fields`) and the
raw-stream rewrite (:mod:`._rewrite`) read these.
"""

from __future__ import annotations

from tokenizer.token_manager import VocabularyManager


_V2_RESERVED_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_V2_VALUE_NEGATIVE_TOKEN_ID = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID  # 256
_V2_NUMBER_BLOCK_START = VocabularyManager._V2_NUMBER_BLOCK_START  # 257
_V2_NUMBER_BLOCK_COUNT = VocabularyManager._V2_NUMBER_BLOCK_COUNT  # 7
_V2_EAGER_BLOCK_END = VocabularyManager._V2_EAGER_BLOCK_END  # 272

_VC2_VOCAB_ID = _V2_NUMBER_BLOCK_START
_FLOAT128_VOCAB_ID = _V2_NUMBER_BLOCK_START + _V2_NUMBER_BLOCK_COUNT - 1
