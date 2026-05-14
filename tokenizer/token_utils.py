import typing
from typing import List, Tuple

import numpy as np
from numpy import typing as npt

from tokenizer.tokens import LitTokenType


class TokenUtils:
    """Utility functions for token caching with reverse mapping"""

    @staticmethod
    def cache_specific_token(
        cls, cache_attr: str, token_string: str, vocab_manager: "VocabularyManager", lit_type: LitTokenType
    ) -> int:
        """Get or create a single cached token with reverse mapping"""
        token_id = None
        if hasattr(cls, cache_attr):
            token_id = getattr(cls, cache_attr)

        if token_id is None:
            token_id = vocab_manager._private_add_token(token_string, cls, lit_type)
            setattr(cls, cache_attr, token_id)

        return token_id

    @staticmethod
    def lookup_specific_token_id(cls, cache_attr: str, token_string: str, vocab_manager: "VocabularyManager") -> int:
        """Get or create a single cached token with reverse mapping"""
        token_id = None
        if hasattr(cls, cache_attr):
            token_id = getattr(cls, cache_attr)

        if token_id is None:
            token_id = vocab_manager.get_token_id(token_string)
            if token_id is None or token_id == -1:
                raise ValueError(f"Token '{token_string}' not found in vocabulary")
            setattr(cls, cache_attr, token_id)

        return token_id

    @staticmethod
    def cache_numeric_token(
        cls, cache_dict_attr: str, key: int, token_id_lambda: typing.Callable[[], int], max_key: int
    ) -> int:
        """Get or create a token from a dictionary cache with numpy array and reverse mapping"""

        # Initialize numpy array if not exists
        if not hasattr(cls, cache_dict_attr):
            setattr(cls, cache_dict_attr, np.full(max_key, -1, dtype=np.int32))

        cache_array = getattr(cls, cache_dict_attr)
        assert len(cache_array) == max_key, "Invalid cache array size"
        assert key < max_key, f"Key {key} exceeds maximum key {max_key}"

        # Get or create token ID
        if cache_array[key] == -1:
            token_id = token_id_lambda()
            if token_id is None or token_id == -1:
                return -1
            cache_array[key] = token_id

            # Add reverse mapping
            reverse_cache_attr = f"{cache_dict_attr}_reverse"
            if not hasattr(cls, reverse_cache_attr):
                setattr(cls, reverse_cache_attr, {})
            getattr(cls, reverse_cache_attr)[token_id] = key

        return cache_array[key]

    @staticmethod
    def cache_numeric_reverse(cls, token_id: int, cache_dict_attr: str, vocab_manager: "VocabularyManager") -> int:
        """Get key from token ID using reverse cache, creating entry if missing"""
        reverse_cache_attr = f"{cache_dict_attr}_reverse"
        # Initialize reverse cache if not exists
        if not hasattr(cls, reverse_cache_attr):
            setattr(cls, reverse_cache_attr, {})

        reverse_cache = getattr(cls, reverse_cache_attr)

        if token_id not in reverse_cache:
            # Get token string and parse hex value after underscore
            token_str = vocab_manager.get_token_str(token_id)
            if token_str is None or token_str == "":
                return -1

            underscore_idx = token_str.rfind("_")
            if underscore_idx != -1:
                hex_part = token_str[underscore_idx + 1 :]
                key = int(hex_part, 16)
            else:
                raise ValueError(f"Cannot parse hex value from token string: {token_str}")

            TokenUtils.cache_numeric_token(
                cls, cache_dict_attr, key, token_id_lambda=lambda: token_id, max_key=16 ** len(hex_part)
            )

        return reverse_cache[token_id]

    """Utility methods for encoding and decoding tokens back to numerical values"""

    @staticmethod
    def encode_tokens(
        basename: str,
        inner_lit_name: str,
        hex_values: npt.NDArray[np.int_],
        vocab_manager: "VocabularyManager",
        token_class,
        inner_token_class,
        max_key: int,
        include_minus: bool = False,
    ) -> List[int]:
        """
        Encode a list of hex values into token IDs using cache system.

        Args:
            basename: The base name for the tokens (e.g., "Block", "VALUED_CONST")
            inner_lit_name: The name for inner tokens (e.g., "Identifier_Lit", "VALUED_CONST")
            hex_values: Numpy array of integer values to encode as hex
            vocab_manager: The vocabulary manager instance
            token_class: The token class to use for caching start/end tokens
            inner_token_class: The token class to use for caching inner tokens
            max_key: Maximum key value for the numpy array cache
            include_minus: Whether to include a minus token (for negative values)

        Returns:
            List of token IDs
        """
        # Calculate required hex digits dynamically
        hex_digits = max_key.bit_length() // 4

        if len(hex_values) == 1 and not include_minus:
            hex_value: int = hex_values[0]
            hex_str = hex(hex_value)[2:].upper()
            hex_str = "0" * (hex_digits - len(hex_str)) + hex_str

            def token_lambda():
                return vocab_manager._private_add_token(f"{basename}_{hex_str}", token_class)

            return [TokenUtils.cache_numeric_token(token_class, f"_{basename}_cache", hex_value, token_lambda, max_key)]
        else:
            # Complex case: multiple tokens with Lit_Start/Lit_End using cache
            token_ids = []

            # Start token - use cache with token_class
            token_ids.append(
                TokenUtils.cache_specific_token(
                    token_class, "_start_token_id", f"{basename}_Lit_Start", vocab_manager, LitTokenType.LIT_START
                )
            )

            # Minus token if needed - use cache
            if include_minus:
                token_ids.extend(vocab_manager.MemoryOperand.MINUS.get_token_ids())

            # Hex value tokens - use updated cache method
            for hex_value in hex_values:
                hex_str = hex(hex_value)[2:].upper()
                hex_str = "0" * (hex_digits - len(hex_str)) + hex_str

                def token_lambda():
                    return vocab_manager._private_add_token(f"{inner_lit_name}_{hex_str}", inner_token_class)

                digit_token_id = TokenUtils.cache_numeric_token(
                    inner_token_class, f"_{inner_lit_name}_cache", hex_value, token_lambda, max_key
                )
                token_ids.append(digit_token_id)

                def token_lambda():
                    return vocab_manager._private_add_token(f"{inner_lit_name}_{hex_str}")

            token_ids.append(
                TokenUtils.cache_specific_token(
                    token_class, "_end_token_id", f"{basename}_Lit_End", vocab_manager, LitTokenType.LIT_END
                )
            )

            return token_ids

    @staticmethod
    def decode_tokens_to_value(
        token_ids: List[int],
        basename: str,
        inner_lit_name: str,
        vocab_manager: "VocabularyManager",
        max_key: int,
        support_negative: bool = False,
        token_class=None,
        inner_token_class=None,
    ) -> int:
        """
        Generic function to decode token IDs back to numerical values.

        Args:
            token_ids: List of token IDs to decode
            basename: The base name for tokens (e.g., "Block", "VALUED_CONST")
            inner_lit_name: The name for inner tokens (e.g., "Identifier_Lit", "VALUED_CONST")
            vocab_manager: The vocabulary manager instance
            max_key: Maximum key value for determining hex format
            support_negative: Whether to check for negative values (MEM_MINUS token)
            token_class: The token class to use for reverse cache lookups (optional)
            inner_token_class: The inner token class to use for reverse cache lookups (optional)

        Returns:
            The decoded integer value
        """
        if len(token_ids) == 1:
            # Simple case: single token
            token_id = token_ids[0]
            hex_value = TokenUtils.cache_numeric_reverse(token_class, token_id, f"_{basename}_cache", vocab_manager)
            return hex_value
        else:
            # Complex case: multiple tokens
            start_found = False
            decoded_value = 0
            shift = max_key.bit_length() - 1
            is_negative = False

            # Get start/end token IDs from cache
            start_token_id = TokenUtils.lookup_specific_token_id(
                token_class, "_start_token_id", f"{basename}_Lit_Start", vocab_manager
            )
            end_token_id = TokenUtils.lookup_specific_token_id(
                token_class, "_end_token_id", f"{basename}_Lit_End", vocab_manager
            )
            minus_token_id = vocab_manager.MemoryOperand.MINUS.get_token_ids() if support_negative else None
            assert minus_token_id is None or len(minus_token_id) == 1, "Only support single minus token"
            minus_token_id = minus_token_id[0] if support_negative else None

            for token_id in token_ids:
                if start_token_id is not None and token_id == start_token_id:
                    start_found = True
                elif end_token_id is not None and token_id == end_token_id:
                    break
                elif minus_token_id is not None and token_id == minus_token_id and start_found:
                    is_negative = True
                elif start_found:
                    # Always use cache function for hex value lookup
                    token_value = TokenUtils.cache_numeric_reverse(
                        inner_token_class, token_id, f"_{inner_lit_name}_cache", vocab_manager
                    )
                    decoded_value <<= shift
                    decoded_value |= token_value

            if is_negative:
                decoded_value = -decoded_value

            return decoded_value

    """V2 wire format: inline-digit encoding.

    In v2 the vocab IDs 0-255 are reserved implicitly as digit tokens (the
    byte value IS the token id). IDs >= 256 are real vocab tokens.
    A token id < 256 means "continuation digit of the previous metatoken";
    >= 256 starts a new metatoken. No Lit_Start/Lit_End framing.
    """

    @staticmethod
    def encode_v2_inline_digits(type_token_id: int, payload: bytes) -> List[int]:
        """V2 inline-digit encoding: returns [type_token_id, *payload_bytes].

        Each payload byte (0-255) IS a digit token id (vocab IDs 0-255 are
        reserved implicitly for digit tokens). No Lit_Start/Lit_End framing.
        """
        assert type_token_id >= 256, f"type_token_id must be >= 256, got {type_token_id}"
        return [type_token_id, *payload]

    @staticmethod
    def decode_v2_inline_digits(token_ids: List[int]) -> Tuple[int, bytes]:
        """V2 inline-digit decoding. Walks ``token_ids``:

          - First element: type_token_id (asserted >= 256).
          - Subsequent: while token_id < 256, append byte to payload; stop at
            first element >= 256 (next metatoken boundary) or end of slice.

        Returns ``(type_token_id, payload_bytes)``. Does NOT consume the
        boundary token.
        """
        assert len(token_ids) >= 1, "token_ids must contain at least the type token"
        type_token_id = token_ids[0]
        assert type_token_id >= 256, f"type_token_id must be >= 256, got {type_token_id}"

        payload = bytearray()
        for token_id in token_ids[1:]:
            if token_id >= 256:
                break
            payload.append(token_id)

        return type_token_id, bytes(payload)

    @staticmethod
    def int_to_minimum_bytes(value: int, *, big_endian: bool = True) -> bytes:
        """Pack non-negative int as minimum-width unsigned bytes.

        ``value=0`` -> ``b'\\x00'`` (1 byte). ``value=255`` -> ``b'\\xff'``.
        ``value=256`` -> ``b'\\x01\\x00'``. Negative values raise ``ValueError``.
        """
        if value < 0:
            raise ValueError(f"int_to_minimum_bytes requires non-negative value, got {value}")
        width = max(1, (value.bit_length() + 7) // 8)
        return value.to_bytes(width, byteorder="big" if big_endian else "little", signed=False)

    @staticmethod
    def bytes_to_int(payload: bytes, *, big_endian: bool = True) -> int:
        """Inverse of :meth:`int_to_minimum_bytes`. Empty bytes raises ``ValueError``."""
        if len(payload) == 0:
            raise ValueError("bytes_to_int requires at least one byte, got empty payload")
        return int.from_bytes(payload, byteorder="big" if big_endian else "little", signed=False)

    @staticmethod
    def encode_v2_float(type_token_id: int, bits: int, width_bytes: int) -> List[int]:
        """V2 float encoding (inline value): ``type_token_id`` followed by exactly
        ``width_bytes`` big-endian bytes of ``bits`` (zero-padded if
        ``bits.bit_length() < 8*width_bytes``).

        Used when the float token carries an inline value (as opposed to acting
        as a postfix type annotation on a preceding ptr token).
        """
        assert type_token_id >= 256, f"type_token_id must be >= 256, got {type_token_id}"
        assert width_bytes >= 1, f"width_bytes must be >= 1, got {width_bytes}"
        if bits < 0:
            raise ValueError(f"encode_v2_float requires non-negative bits, got {bits}")
        if bits.bit_length() > 8 * width_bytes:
            raise ValueError(
                f"bits 0x{bits:x} does not fit in {width_bytes} bytes ({bits.bit_length()} bits)"
            )
        payload = bits.to_bytes(width_bytes, byteorder="big", signed=False)
        return [type_token_id, *payload]

    @staticmethod
    def encode_v2_float_postfix(type_token_id: int) -> List[int]:
        """V2 float postfix annotation: just ``[type_token_id]``.

        The next token (>= 256) starts the next metatoken; no inline value
        bytes. Used to annotate a preceding ptr token with the FP type loaded
        from the pointed-to address (e.g. ``movss [ro_data_ptr], xmm0``).
        """
        assert type_token_id >= 256, f"type_token_id must be >= 256, got {type_token_id}"
        return [type_token_id]
