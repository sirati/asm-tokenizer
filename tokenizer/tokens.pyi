from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Optional, Type, TypeVar

import numpy as np
import numpy.typing as npt

from tokenizer.architecture import PlatformInstructionTypes

T = TypeVar("T", bound="Tokens")

class TokenType(IntEnum):
    ERROR: "TokenType"
    PLATFORM: "TokenType"
    VALUED_CONST: "TokenType"
    BLOCK_DEF: "TokenType"
    BLOCK: "TokenType"
    OPAQUE_CONST: "TokenType"
    MEMORY_OPERAND: "TokenType"
    TOKEN_SET: "TokenType"
    IDENTIFIER_LITERAL: "TokenType"
    LOCAL_FUNCTION: "TokenType"
    LOCAL_FUNC: "TokenType"
    PLT_FUNC: "TokenType"
    EXT_FUNC: "TokenType"
    RO_DATA_PTR: "TokenType"
    RW_DATA_PTR: "TokenType"
    STRING_PTR: "TokenType"
    JUMP_TABLE: "TokenType"
    VALUED_CONST_V2: "TokenType"
    BLOCK_V2: "TokenType"
    FLOAT16: "TokenType"
    BFLOAT16: "TokenType"
    FLOAT32: "TokenType"
    FLOAT64: "TokenType"
    FLOAT80: "TokenType"
    FLOAT128: "TokenType"
    THREAD_LOCAL: "TokenType"
    VTABLE: "TokenType"
    CODE_PTR_TABLE: "TokenType"
    VARIANT_AXIS: "TokenType"
    UNRESOLVED: "TokenType"

class MemoryOperandSymbol(Enum):
    OPEN_BRACKET: str
    CLOSE_BRACKET: str
    PLUS: str
    MINUS: str
    MULTIPLY: str
    def token_str(self) -> str: ...

class Tokens(ABC):
    @property
    @classmethod
    @abstractmethod
    def token_type(cls) -> TokenType: ...
    @classmethod
    @abstractmethod
    def _from_token_ids(cls, token_ids: List[int]) -> "Tokens": ...
    @abstractmethod
    def get_token_ids(self) -> npt.NDArray[np.int_]: ...
    @abstractmethod
    def to_string(self) -> str: ...
    @abstractmethod
    def to_asm_like(self) -> str: ...
    @property
    def platform_instruction_type(self) -> PlatformInstructionTypes: ...
    def register_on_vocab_manager(self, other: "VocabularyManager") -> Tokens: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other) -> bool: ...

class PlatformToken(Tokens, ABC):
    token: str
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @property
    @abstractmethod
    def platform(self) -> str: ...
    @property
    @abstractmethod
    def platform_instruction_type(self) -> PlatformInstructionTypes: ...
    @abstractmethod
    def __init__(self, token: str, insn_type: PlatformInstructionTypes) -> None: ...

class ValuedConstToken(Tokens, ABC):
    value: int
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, value: int) -> None: ...

class IdentifierToken(Tokens, ABC):
    id: int
    @abstractmethod
    def __init__(self, identifier_id: int) -> None: ...
    @classmethod
    @abstractmethod
    def _get_basename(cls) -> str: ...

class BlockDefToken(Tokens, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self) -> None: ...

class BlockToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, block_id: int) -> None: ...

class OpaqueConstToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, opaque_id: int) -> None: ...

class LocalFunctionToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, local_function_id: int) -> None: ...

# v2 identity-carrying category tokens (see plan vivid-tinkering-wilkes.md)
class LocalFuncToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, local_func_id: int) -> None: ...

class PltFuncToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, plt_func_id: int) -> None: ...

class ExtFuncToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, ext_func_id: int) -> None: ...

class RoDataPtrToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, ro_data_ptr_id: int) -> None: ...

class RwDataPtrToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, rw_data_ptr_id: int) -> None: ...

class StringPtrToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, string_ptr_id: int) -> None: ...

class JumpTableToken(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, jump_table_id: int) -> None: ...

class BlockTokenV2(IdentifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, block_id: int) -> None: ...

class ValuedConstTokenV2(ValuedConstToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, value: int) -> None: ...

# v2 float tokens — dual-mode: inline-value when `bits` is set, postfix
# annotation when `bits is None`. The reader distinguishes them by the
# next-token-< 256 rule, not by the in-memory type.
class FloatToken(Tokens, ABC):
    width_bytes: int
    bits: Optional[int]
    @abstractmethod
    def __init__(self, bits: Optional[int] = ...) -> None: ...

class Float16Token(FloatToken, ABC):
    width_bytes: int
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, bits: Optional[int] = ...) -> None: ...

class BFloat16Token(FloatToken, ABC):
    width_bytes: int
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, bits: Optional[int] = ...) -> None: ...

class Float32Token(FloatToken, ABC):
    width_bytes: int
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, bits: Optional[int] = ...) -> None: ...

class Float64Token(FloatToken, ABC):
    width_bytes: int
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, bits: Optional[int] = ...) -> None: ...

class Float80Token(FloatToken, ABC):
    width_bytes: int
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, bits: Optional[int] = ...) -> None: ...

class Float128Token(FloatToken, ABC):
    width_bytes: int
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, bits: Optional[int] = ...) -> None: ...

# v2 modifier tokens — no identity, no payload; precede a base category token.
class ModifierToken(Tokens, ABC):
    @abstractmethod
    def __init__(self) -> None: ...

class ThreadLocalToken(ModifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self) -> None: ...

class VtableToken(ModifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self) -> None: ...

class CodePtrTableToken(ModifierToken, ABC):
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self) -> None: ...

class VariantAxisToken(Tokens, ABC):
    token: str
    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @abstractmethod
    def __init__(self, token: str) -> None: ...

class MemoryOperandToken(Tokens, ABC):
    symbol: MemoryOperandSymbol

    # This dataclass is created by the decorator as a class member
    @dataclass
    class EnumTokenCache:
        OPEN_BRACKET: "MemoryOperandToken | None" = None
        CLOSE_BRACKET: "MemoryOperandToken | None" = None
        PLUS: "MemoryOperandToken | None" = None
        MINUS: "MemoryOperandToken | None" = None
        MULTIPLY: "MemoryOperandToken | None" = None

    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    @classmethod
    @abstractmethod
    def _get_enum_token_cache(cls) -> EnumTokenCache: ...
    @classmethod
    @abstractmethod
    def _from_enum(cls, symbol: MemoryOperandSymbol) -> "MemoryOperandToken": ...

    # Class properties that will be created by the decorator
    @classmethod
    @property
    def OPEN_BRACKET(cls) -> "MemoryOperandToken": ...
    @classmethod
    @property
    def CLOSE_BRACKET(cls) -> "MemoryOperandToken": ...
    @classmethod
    @property
    def PLUS(cls) -> "MemoryOperandToken": ...
    @classmethod
    @property
    def MINUS(cls) -> "MemoryOperandToken": ...
    @classmethod
    @property
    def MULTIPLY(cls) -> "MemoryOperandToken": ...
    @abstractmethod
    def __init__(self, symbol: MemoryOperandSymbol) -> None: ...

class Category(Enum):
    BLOCK: "Category"
    LOCAL_FUNC: "Category"
    PLT_FUNC: "Category"
    EXT_FUNC: "Category"
    RO_DATA_PTR: "Category"
    RW_DATA_PTR: "Category"
    STRING_PTR: "Category"
    JUMP_TABLE: "Category"

class TokenResolver:
    counters: dict[Category, int]
    id_maps: dict[Category, dict[int, int]]
    metadata: dict[Category, list[dict]]

    # v1-only opaque state
    opaque_counter: int
    opaque_ids: dict[int, int]

    # v1-backed read-only views over BLOCK / LOCAL_FUNC category state
    @property
    def block_counter(self) -> int: ...
    @property
    def block_ids(self) -> dict[int, int]: ...
    @property
    def local_function_counter(self) -> int: ...
    @property
    def local_function_ids(self) -> dict[int, int]: ...

    def __init__(self) -> None: ...
    def get_identity(
        self, category: Category, addr: int, meta: Optional[dict] = ...
    ) -> int: ...
    def reset_function(self) -> None: ...
    def get_block_id(self, addr: int) -> int: ...
    def get_opaque_id(self, addr: int) -> int: ...
    def get_local_function_id(self, addr: int) -> int: ...
    def reset(self) -> None: ...

def EnumTokenCls(enum_class: Type[Enum]) -> Type[T]: ...

class TokenRaw(Tokens):
    token_ids_array: npt.NDArray[np.int_]
    token_type_enum: TokenType

    @property
    @classmethod
    def token_type(cls) -> TokenType: ...
    def resolve(self, vocab_manager: "VocabularyManager") -> "Tokens": ...
    @staticmethod
    def with_type(token_type_enum: TokenType) -> type["TokenRaw"]: ...

class LitTokenType(Enum):
    REGULAR: "LitTokenType"
    LIT_START: "LitTokenType"
    LIT_END: "LitTokenType"
    LIT_END: "LitTokenType"
