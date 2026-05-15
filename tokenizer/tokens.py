from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, List, Optional, Type, TypeVar

import numpy as np
import numpy.typing as npt

from tokenizer.architecture import PlatformInstructionTypes

T = TypeVar("T", bound="Tokens")


def EnumTokenCls(enum_class: Type[Enum]) -> Any:
    """Decorator to create lazy class properties for all enum members and required infrastructure"""

    def decorator(cls: Type[T]) -> Type[T]:
        # Get the enum name and create dataclass name
        enum_name = enum_class.__name__
        dataclass_name = "EnumTokenCache"

        # Create the dataclass dynamically
        dataclass_fields = {}
        for member in enum_class:
            dataclass_fields[member.name] = "MemoryOperandToken | None"

        # Create the dataclass type
        dataclass_type = type(
            dataclass_name,
            (),
            {
                "__annotations__": {name: typ for name, typ in dataclass_fields.items()},
                "__module__": cls.__module__,
                "__doc__": f"Dataclass containing all {enum_name.lower()} symbol tokens",
                **{name: None for name in dataclass_fields.keys()},
            },
        )
        # Apply dataclass decorator with type ignore for the warning
        dataclass_type = dataclass(dataclass_type)  # type: ignore

        # Add the dataclass as a class member instead of module globals
        setattr(cls, dataclass_name, dataclass_type)

        # Add abstract methods to the class
        def _get_enum_token_cache(cls):
            """Return a dataclass instance containing all symbol tokens"""
            pass

        _get_enum_token_cache.__doc__ = f"Return a dataclass instance containing all {enum_name.lower()} symbol tokens"

        def _from_enum(cls, symbol):
            """Create token from enum member"""
            pass

        _from_enum.__doc__ = f"Create token from {enum_name} member"

        # Make them abstract methods
        _get_enum_token_cache = classmethod(abstractmethod(_get_enum_token_cache))
        _from_enum = classmethod(abstractmethod(_from_enum))

        # Add to class
        setattr(cls, "_get_enum_token_cache", _get_enum_token_cache)
        setattr(cls, "_from_enum", _from_enum)

        # Create properties for each enum member
        for member in enum_class:
            property_name = member.name

            # Create the property method
            def create_property_method(enum_member):
                def property_method(cls):
                    syms = cls._get_enum_token_cache()
                    attr_name = enum_member.name
                    if getattr(syms, attr_name) is None:
                        setattr(syms, attr_name, cls._from_enum(enum_member))
                    return getattr(syms, attr_name)

                return property_method

            property_method = create_property_method(member)
            property_method.__doc__ = (
                f"Return a {enum_name.lower()} token for {property_name.lower().replace('_', ' ')} symbol"
            )

            # Create a class property using a descriptor approach
            class ClassPropertyDescriptor:
                def __init__(self, func):
                    self.func = func
                    self.__doc__ = func.__doc__

                def __get__(self, obj, cls):
                    return self.func(cls)

            # Add as class property
            setattr(cls, property_name, ClassPropertyDescriptor(property_method))

        return cls

    return decorator


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class TokenType(IntEnum):
    """Enum for token types to identify token classes"""

    ERROR = 0
    PLATFORM = 1
    VALUED_CONST = 2
    BLOCK_DEF = 3
    BLOCK = 4
    OPAQUE_CONST = 5
    MEMORY_OPERAND = 6
    TOKEN_SET = 7
    IDENTIFIER_LITERAL = 8
    LOCAL_FUNCTION = 9
    # v2 category tokens — see plan vivid-tinkering-wilkes.md "Target taxonomy".
    # Identity- or value-carrying categories whose payload (if any) rides
    # inline as digit tokens at vocab IDs 0–255 after the type-id.
    LOCAL_FUNC = 10
    PLT_FUNC = 11
    EXT_FUNC = 12
    RO_DATA_PTR = 13
    RW_DATA_PTR = 14
    STRING_PTR = 15
    JUMP_TABLE = 16
    VALUED_CONST_V2 = 17
    BLOCK_V2 = 18
    # v2 float tokens — fixed-width payloads (in bytes: 2, 2, 4, 8, 10, 16).
    # Dual-mode: inline-value (digit tokens follow) OR postfix annotation
    # (no digits follow; annotates the preceding ptr token's load type).
    FLOAT16 = 19
    BFLOAT16 = 20
    FLOAT32 = 21
    FLOAT64 = 22
    FLOAT80 = 23
    FLOAT128 = 24
    # v2 modifiers — no identity, no payload; precede a base category token.
    THREAD_LOCAL = 25
    VTABLE = 26
    CODE_PTR_TABLE = 27
    # ARM register-list bracketing (stm/ldm/push/pop/vpush/vpop/vstm/vldm).
    # Parallel to MEMORY_OPERAND in shape: a small fixed enum of bracket /
    # writeback symbols, paired open/close framing around the register-token
    # members.
    REGISTER_LIST = 28
    UNRESOLVED = -1


class MemoryOperandSymbol(Enum):
    """Enum for memory operand symbols"""

    OPEN_BRACKET = "mem["
    CLOSE_BRACKET = "]mem"
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"

    def token_str(self) -> str:
        """Get the string representation of the memory operand symbol"""
        if self == MemoryOperandSymbol.OPEN_BRACKET:
            return "MEM_OPEN_BRACKET"
        elif self == MemoryOperandSymbol.CLOSE_BRACKET:
            return "MEM_CLOSE_BRACKET"
        elif self == MemoryOperandSymbol.PLUS:
            return "MEM_PLUS"
        elif self == MemoryOperandSymbol.MINUS:
            return "MEM_MINUS"
        elif self == MemoryOperandSymbol.MULTIPLY:
            return "MEM_MULTIPLY"
        else:
            raise ValueError(f"Unknown memory operand symbol: {self}")


class RegisterListSymbol(Enum):
    """Enum for ARM register-list bracketing + writeback symbols.

    Mirrors the real ARM asm syntax ``stmdb sp!, {r4, lr}``: open/close
    braces frame the list members; the writeback ``!`` marks the base
    register as updated. The asm-form values are used as the asm-like
    debug string; ``token_str()`` returns the vocab-stable name.
    """

    OPEN_BRACE = "reglist{"
    CLOSE_BRACE = "}reglist"
    WRITEBACK = "!"

    def token_str(self) -> str:
        """Get the vocab-stable string for the register-list symbol."""
        if self == RegisterListSymbol.OPEN_BRACE:
            return "REG_LIST_OPEN_BRACE"
        elif self == RegisterListSymbol.CLOSE_BRACE:
            return "REG_LIST_CLOSE_BRACE"
        elif self == RegisterListSymbol.WRITEBACK:
            # Vocab name is ISA-neutral and uniquely identifies the
            # writeback semantic (sp != sp!, but the register itself
            # stays `arm32_sp` for both -- the writeback is a separate
            # marker token). The enum value `"!"` above is the
            # to_asm_like render; this string is the vocab identifier.
            return "asm_writeback_detect"
        else:
            raise ValueError(f"Unknown register-list symbol: {self}")


class Tokens(ABC):
    """Protocol for token representation objects"""

    @classproperty
    @abstractmethod
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        ...

    @classmethod
    @abstractmethod
    def _from_token_ids(cls, token_ids: List[int]) -> "Tokens": ...

    @abstractmethod
    def get_token_ids(self) -> npt.NDArray[np.int_]:
        """Get the list of token IDs for this token representation (order matters)"""
        ...

    @abstractmethod
    def to_string(self) -> str:
        """Convert token to its string representation (for debugging only)"""
        ...

    def register_on_vocab_manager(self, other: "VocabularyManager") -> "Tokens":
        return self._register_on(other.get_token_class_for_type(self.token_type))

    @abstractmethod
    def _register_on(self, other_cls: type["Tokens"]) -> "Tokens": ...

    @abstractmethod
    def to_asm_like(self) -> str:
        """Convert token to its string representation that resembles assembly syntax"""
        ...

    @property
    def platform_instruction_type(self) -> PlatformInstructionTypes:
        """Return the platform instruction type for this token"""
        return PlatformInstructionTypes.AGNOSTIC

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__.replace('Inner', 'Token').replace('TokenToken', 'Token')}({self.to_string()})"

    def __hash__(self) -> int:
        """Make tokens hashable based on class and token IDs"""
        return hash((self.__class__.__name__, tuple(self.get_token_ids())))

    def __eq__(self, other) -> bool:
        """Tokens are equal if they have the same class and same token IDs"""
        if (not isinstance(other, Tokens)) or self.token_type != other.token_type:
            return False
        myids = self.get_token_ids()
        otherids = other.get_token_ids()

        return myids.shape == otherids.shape and np.all(myids == otherids)


class PlatformToken(Tokens, ABC):
    """Protocol for platform-specific tokens"""

    token: str

    @classproperty
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        return TokenType.PLATFORM

    @property
    @abstractmethod
    def platform(self) -> str: ...

    @property
    @abstractmethod
    def platform_instruction_type(self) -> PlatformInstructionTypes: ...

    @abstractmethod
    def __init__(self, token: str, insn_type: PlatformInstructionTypes, platform: str = None) -> None: ...

    def _register_on(self, cls_other):
        return cls_other(self.token, self.platform_instruction_type, self.platform)


class ValuedConstToken(Tokens, ABC):
    """Protocol for valued constants"""

    @classproperty
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        return TokenType.VALUED_CONST

    value: int

    @abstractmethod
    def __init__(self, value: int) -> None: ...

    def _register_on(self, cls_other):
        return cls_other(self.value)


class IdentifierToken(Tokens, ABC):
    """Protocol for identifier tokens"""

    id: int

    @abstractmethod
    def __init__(self, identifier_id: int) -> None: ...

    @classproperty
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        return TokenType.IDENTIFIER_LITERAL

    @classmethod
    def singleton_token_index(cls, id: int) -> Optional[int]:
        """
        Get the index of this identifier token in the vocabulary. If the identifier can be represented as a singleton token,


        Returns:
            Index of this identifier token in the vocabulary
        """
        ...

    @classmethod
    def value_by_singleton_token_index(cls, index: int) -> Optional[int]:
        """
        Get the value of this identifier token by its index in the vocabulary. If the identifier can be represented as a singleton token,


        Args:
            index: Index of the identifier token in the vocabulary

        Returns:
            Value of this identifier token
        """
        ...

    @classmethod
    @abstractmethod
    def _get_basename(cls) -> str:
        """Get the base name for this identifier type"""
        ...

    def _register_on(self, cls_other):
        return cls_other(self.id)


class BlockDefToken(Tokens, ABC):
    """Protocol for block definition tokens"""

    @classproperty
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        return TokenType.BLOCK_DEF

    @abstractmethod
    def __init__(self) -> None: ...

    def _register_on(self, cls_other):
        return cls_other()


class BlockToken(IdentifierToken, ABC):
    """Protocol for block identifiers"""

    @classproperty
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        return TokenType.BLOCK

    @abstractmethod
    def __init__(self, block_id: int) -> None: ...


class OpaqueConstToken(IdentifierToken, ABC):
    """Protocol for opaque constants"""

    @classproperty
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        return TokenType.OPAQUE_CONST

    @abstractmethod
    def __init__(self, opaque_id: int) -> None: ...


class LocalFunctionToken(IdentifierToken, ABC):
    """Protocol for local function identifiers - used by inlining matcher, not by tokenizer"""

    @classproperty
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        return TokenType.LOCAL_FUNCTION

    @abstractmethod
    def __init__(self, local_function_id: int) -> None: ...


# ---------------------------------------------------------------------------
# v2 token classes (plan vivid-tinkering-wilkes.md)
#
# Identity-carrying category tokens reuse the existing IdentifierToken ABC
# (id: int + _register_on -> cls_other(self.id) + _get_basename) and only
# override `token_type` and `_get_basename` per category. This honors the
# project rule against duplicated logic: the inline-digit encoding for a v2
# identity is the same shape as the legacy IdentifierToken digit framing —
# only the vocab basename and TokenType differ. Concrete *Inner subclasses
# (added separately in token_manager.py) supply the per-category encoding
# path; these ABCs are stubs that establish the type hierarchy + register
# the TokenType.
# ---------------------------------------------------------------------------


class LocalFuncToken(IdentifierToken, ABC):
    """Protocol for v2 local-function category tokens (function entry in main object).

    Identity is per-function: the i-th distinct local function encountered in
    a function body gets id i. Decomposable=no; the pointer is to executable
    code in the main object and has no further structure.
    """

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.LOCAL_FUNC

    @abstractmethod
    def __init__(self, local_func_id: int) -> None: ...


class PltFuncToken(IdentifierToken, ABC):
    """Protocol for v2 PLT-stub category tokens (PLT entry in any loaded object)."""

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.PLT_FUNC

    @abstractmethod
    def __init__(self, plt_func_id: int) -> None: ...


class ExtFuncToken(IdentifierToken, ABC):
    """Protocol for v2 external-function category tokens.

    Covers both real function entries in other loaded objects AND CLE
    synthetic extern slots; provider metadata distinguishes the two via the
    `synthetic` flag, but the token itself is identity-only.
    """

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.EXT_FUNC

    @abstractmethod
    def __init__(self, ext_func_id: int) -> None: ...


class RoDataPtrToken(IdentifierToken, ABC):
    """Protocol for v2 pointer-into-rodata category tokens.

    Decomposable=yes; emitters may decompose into `[ro_data_ptr_base + valued_const_offset]`
    when an exact slot identity isn't available.
    """

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.RO_DATA_PTR

    @abstractmethod
    def __init__(self, ro_data_ptr_id: int) -> None: ...


class RwDataPtrToken(IdentifierToken, ABC):
    """Protocol for v2 pointer-into-rw-data category tokens.

    Covers `.data`, `.bss`, `.tdata`, `.tbss` — TLS sections additionally get
    a leading `ThreadLocalToken` modifier. Decomposable=yes.
    """

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.RW_DATA_PTR

    @abstractmethod
    def __init__(self, rw_data_ptr_id: int) -> None: ...


class StringPtrToken(IdentifierToken, ABC):
    """Protocol for v2 string-pointer category tokens.

    Identity refers to a provider-confirmed string; the string content lives
    in the per-binary `<binary>_strings.bin` sidecar, addressed via the
    function's `metadata.string_ptr[id] = {line, start_offset, encoding}`.
    Decomposable=yes (substring access at non-zero start_offset).
    """

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.STRING_PTR

    @abstractmethod
    def __init__(self, string_ptr_id: int) -> None: ...


class JumpTableToken(IdentifierToken, ABC):
    """Protocol for v2 jump-table category tokens (switch-statement target arrays).

    Identity is per-function; the table's definition (slot block-ids in slot
    order) is appended to the function's footer as a `block_def jump_table <id>`
    region. Decomposable=no.
    """

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.JUMP_TABLE

    @abstractmethod
    def __init__(self, jump_table_id: int) -> None: ...


class BlockTokenV2(IdentifierToken, ABC):
    """Protocol for v2 block identifiers.

    Distinct from the legacy `BlockToken` (TokenType.BLOCK) so the v1 emission
    path and its vocab entries (`Block_0..N`) remain valid for class-stability,
    while v2 emits a single `block` vocab entry whose identity rides as inline
    digits.
    """

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.BLOCK_V2

    @abstractmethod
    def __init__(self, block_id: int) -> None: ...


class ValuedConstTokenV2(ValuedConstToken, ABC):
    """Protocol for v2 valued-constant tokens.

    Same `value: int` shape as the legacy `ValuedConstToken`, but the wire
    encoding is the v2 inline-digit form: one type-id followed by N variable-
    width bytes (1..64) of the value, MSB-first. Distinct from legacy
    `ValuedConstToken` to keep v1 vocab entries (`VALUED_CONST_00..FF`,
    Lit_Start / Lit_End framing) untouched.
    """

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.VALUED_CONST_V2

    @abstractmethod
    def __init__(self, value: int) -> None: ...


class FloatToken(Tokens, ABC):
    """Protocol common to all v2 float tokens (16 / bfloat16 / 32 / 64 / 80 / 128).

    Dual-mode payload:
      * Inline-value form: `bits` holds the IEEE bit pattern as an unsigned
        int; the wire stream is `[type_id, *byte_to_digit(b) for b in bits.to_bytes(width_bytes, "big")]`.
      * Postfix-annotation form: `bits is None`; the float token immediately
        follows a pointer token (e.g., `ro_data_ptr <id> float32`) and carries
        no inline digits — the reader's "next token >= 256" check distinguishes
        the two forms.

    Each width is a separate concrete subclass with a `width_bytes` classvar
    so the reader can pick the right consumption length without a runtime
    switch on `token_type`.
    """

    width_bytes: int
    bits: Optional[int]

    @abstractmethod
    def __init__(self, bits: Optional[int] = None) -> None: ...

    def _register_on(self, cls_other):
        return cls_other(self.bits)


class Float16Token(FloatToken, ABC):
    """Protocol for v2 IEEE-754 half (2-byte) float tokens."""

    width_bytes = 2

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.FLOAT16

    @abstractmethod
    def __init__(self, bits: Optional[int] = None) -> None: ...


class BFloat16Token(FloatToken, ABC):
    """Protocol for v2 Google brain-float (2-byte) tokens.

    Distinct enum + class from `Float16Token` because the bit layout differs
    (8-bit exponent + 7-bit mantissa vs. 5-bit exponent + 10-bit mantissa);
    a reader must know which to apply.
    """

    width_bytes = 2

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.BFLOAT16

    @abstractmethod
    def __init__(self, bits: Optional[int] = None) -> None: ...


class Float32Token(FloatToken, ABC):
    """Protocol for v2 IEEE-754 single (4-byte) float tokens."""

    width_bytes = 4

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.FLOAT32

    @abstractmethod
    def __init__(self, bits: Optional[int] = None) -> None: ...


class Float64Token(FloatToken, ABC):
    """Protocol for v2 IEEE-754 double (8-byte) float tokens."""

    width_bytes = 8

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.FLOAT64

    @abstractmethod
    def __init__(self, bits: Optional[int] = None) -> None: ...


class Float80Token(FloatToken, ABC):
    """Protocol for v2 x87 extended-precision (10-byte) float tokens.

    No IEEE-754 standard width matches; the bit pattern is the raw x87 80-bit
    layout (1 sign + 15 exponent + 64 significand including explicit integer
    bit). Stored MSB-first as 10 inline digit bytes.
    """

    width_bytes = 10

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.FLOAT80

    @abstractmethod
    def __init__(self, bits: Optional[int] = None) -> None: ...


class Float128Token(FloatToken, ABC):
    """Protocol for v2 IEEE-754 quad (16-byte) float tokens."""

    width_bytes = 16

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.FLOAT128

    @abstractmethod
    def __init__(self, bits: Optional[int] = None) -> None: ...


class ModifierToken(Tokens, ABC):
    """Protocol common to v2 modifier tokens.

    Modifiers carry no identity and no payload — they're parameterless category
    markers that precede a base category token to refine its meaning. Multiple
    modifiers may stack (e.g., `thread_local rw_data_ptr <id>`). The wire form
    is exactly one vocab id and no inline digits.
    """

    @abstractmethod
    def __init__(self) -> None: ...

    def _register_on(self, cls_other):
        return cls_other()


class ThreadLocalToken(ModifierToken, ABC):
    """Protocol for the v2 `thread_local` modifier (TLS access marker)."""

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.THREAD_LOCAL

    @abstractmethod
    def __init__(self) -> None: ...


class VtableToken(ModifierToken, ABC):
    """Protocol for the v2 `vtable` modifier (RTTI-confirmed C++ vtable slot)."""

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.VTABLE

    @abstractmethod
    def __init__(self) -> None: ...


class CodePtrTableToken(ModifierToken, ABC):
    """Protocol for the v2 `code_ptr_table` modifier.

    Covers non-vtable function-pointer-array slots: `.init_array`, `.fini_array`,
    `.dtors`, dispatch tables, etc.
    """

    @classproperty
    def token_type(cls) -> TokenType:
        return TokenType.CODE_PTR_TABLE

    @abstractmethod
    def __init__(self) -> None: ...


@EnumTokenCls(MemoryOperandSymbol)
class MemoryOperandToken(Tokens, ABC):
    """Protocol for memory operand symbol tokens"""

    @classproperty
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        return TokenType.MEMORY_OPERAND

    symbol: MemoryOperandSymbol

    @abstractmethod
    def __init__(self, symbol: MemoryOperandSymbol) -> None: ...

    def _register_on(self, cls_other):
        return cls_other(self.symbol)


@EnumTokenCls(RegisterListSymbol)
class RegisterListToken(Tokens, ABC):
    """Protocol for ARM register-list bracketing tokens.

    Parallel shape to ``MemoryOperandToken``: a fixed enum of symbols
    (OPEN_BRACE / CLOSE_BRACE / WRITEBACK), one vocab entry per symbol,
    framing inline register tokens for stm/ldm-family ARM instructions.
    """

    @classproperty
    def token_type(cls) -> TokenType:
        """Return the type of this token representation"""
        return TokenType.REGISTER_LIST

    symbol: RegisterListSymbol

    @abstractmethod
    def __init__(self, symbol: RegisterListSymbol) -> None: ...

    def _register_on(self, cls_other):
        return cls_other(self.symbol)


class TokenRaw(Tokens, ABC):
    _cache: dict[TokenType, type["TokenRaw"]] = {}

    @abstractmethod
    def resolve(self, vocab_manager: "VocabularyManager") -> "Tokens": ...

    def _register_on(self, cls_other):
        raise NotImplementedError("TokenRaw cannot be registered directly, please resolve first!")

    @staticmethod
    def with_type(token_type_enum: TokenType) -> type["TokenRaw"]:
        """
        Create a new TokenRaw with the specified token type.

        Args:
            token_type_enum: TokenType enum value to set for the new token

        Returns:
            New TokenRaw instance with the specified type
        """

        if token_type_enum in TokenRaw._cache:
            return TokenRaw._cache[token_type_enum]
        else:

            class TokenRawInner(TokenRaw):
                """Raw token representation with numpy array of IDs and token type"""

                def __init__(self, token_ids: npt.NDArray[np.int_]):
                    """
                    Initialize TokenRaw with token IDs and type

                    Args:
                        token_ids: Numpy array of token IDs
                        token_type_enum: TokenType enum value
                    """
                    super().__init__()
                    self.token_ids_array = token_ids
                    if len(token_ids) == 0:
                        raise ValueError("TokenRaw must have at least one token ID")

                @classproperty
                def token_type(cls) -> TokenType:
                    """Return the type of this token representation"""
                    return token_type_enum

                @classmethod
                def _from_token_ids(cls, token_ids: List[int]) -> "TokenRawInner":
                    """Create TokenRaw from token IDs - type must be determined from context"""
                    return cls(np.array(token_ids, dtype=np.int_))

                def get_token_ids(self) -> npt.NDArray[np.int_]:
                    """Get the list of token IDs for this token representation"""
                    return self.token_ids_array

                def to_string(self) -> str:
                    """Convert token to its string representation (for debugging only)"""
                    return f"TokenRaw({token_type_enum.name}, {self.token_ids_array.tolist()})"

                def to_asm_like(self) -> str:
                    """Convert token to its string representation that resembles assembly syntax"""
                    return f"raw:{token_type_enum.name}:{','.join(map(str, self.token_ids_array))}"

                def resolve(self, vocab_manager: "VocabularyManager") -> "Tokens":
                    """
                    Resolve this TokenRaw into a concrete token using the VocabularyManager

                    Args:
                        vocab_manager: VocabularyManager instance to resolve token IDs

                    Returns:
                        Concrete token instance based on the token type and IDs
                    """
                    token_ids_list = self.get_token_ids()
                    return vocab_manager._reconstruct_token_from_ids(token_type_enum, token_ids_list)

            TokenRaw._cache[token_type_enum] = TokenRawInner
            return TokenRawInner


class Category(Enum):
    """v2 token categories that own a per-function identity space.

    Each member corresponds to a `TokenType` whose wire form carries an
    integer identity (counter-allocated, optionally address-cached, with
    parallel human-readable metadata). Modifier tokens (`thread_local`,
    `vtable`, `code_ptr_table`), `valued_const`, and `floatXX` deliberately
    do NOT appear here -- they have no identity (they ride as either inline
    digits or as fixed pure-type tokens prefixing the following identity).
    """

    BLOCK = "block"
    LOCAL_FUNC = "local_func"
    PLT_FUNC = "plt_func"
    EXT_FUNC = "ext_func"
    RO_DATA_PTR = "ro_data_ptr"
    RW_DATA_PTR = "rw_data_ptr"
    STRING_PTR = "string_ptr"
    JUMP_TABLE = "jump_table"


class TokenResolver:
    """Manages per-function identity allocation for v2 categories.

    Internal state is dict-keyed by `Category`:

    - ``counters[c]``: next-id-to-allocate for category ``c``
    - ``id_maps[c]``: address -> identity cache for category ``c``
    - ``metadata[c]``: identity-indexed list of human-readable metadata dicts

    `get_identity(category, addr, meta=None)` is the v2 entry point. The v1
    methods `get_block_id`, `get_opaque_id`, `get_local_function_id`, and
    `reset` remain available for legacy callers; `get_block_id` and
    `get_local_function_id` are thin shims over the BLOCK/LOCAL_FUNC
    categories (state is shared), while `get_opaque_id` keeps its own
    counter/dict because v1 "opaque" doesn't map cleanly to a single v2
    category (it splits across `ro_data_ptr`/`rw_data_ptr`/`string_ptr`/
    `valued_const`).
    """

    def __init__(self):
        self.counters: dict[Category, int] = {c: 0 for c in Category}
        self.id_maps: dict[Category, dict[int, int]] = {c: {} for c in Category}
        self.metadata: dict[Category, list[dict]] = {c: [] for c in Category}

        # v1-only opaque state -- intentionally separate from any `Category`
        # because v1 "opaque" maps to multiple v2 categories. Removed once
        # all v1 call sites are migrated to `get_identity(...)`.
        self.opaque_counter = 0
        self.opaque_ids: dict[int, int] = {}

    def get_identity(
        self, category: Category, addr: int, meta: Optional[dict] = None
    ) -> int:
        """Get or allocate an identity within `category`.

        If `addr` is truthy and already cached for this category, the cached
        identity is returned (idempotent; `meta` is ignored on a cache hit).
        Otherwise a fresh identity is allocated from `counters[category]`,
        cached under `addr` (when truthy), and `meta or {}` is appended to
        `metadata[category]` so that `metadata[category][identity]` is the
        description for that identity.

        Passing `addr=0` (or any falsy value) allocates an identity without
        caching, preserving v1 semantics for "no address available".
        """
        cache = self.id_maps[category]
        if addr and addr in cache:
            return cache[addr]

        identity = self.counters[category]
        if addr:
            cache[addr] = identity
        self.counters[category] += 1
        self.metadata[category].append(meta or {})
        return identity

    def reset_function(self):
        """Zero all per-category counters and clear maps + metadata.

        Invoked at function boundary so identities restart from 0 each
        function (matching v1's per-function reset semantics).
        """
        for c in Category:
            self.counters[c] = 0
            self.id_maps[c].clear()
            self.metadata[c].clear()

    # ------------------------------------------------------------------
    # v1 backward-compatibility surface
    # ------------------------------------------------------------------

    @property
    def block_counter(self) -> int:
        return self.counters[Category.BLOCK]

    @property
    def block_ids(self) -> dict[int, int]:
        return self.id_maps[Category.BLOCK]

    @property
    def local_function_counter(self) -> int:
        return self.counters[Category.LOCAL_FUNC]

    @property
    def local_function_ids(self) -> dict[int, int]:
        return self.id_maps[Category.LOCAL_FUNC]

    def get_block_id(self, addr: int) -> int:
        """v1 shim: allocate within `Category.BLOCK`."""
        return self.get_identity(Category.BLOCK, addr)

    def get_local_function_id(self, addr: int) -> int:
        """v1 shim: allocate within `Category.LOCAL_FUNC`."""
        return self.get_identity(Category.LOCAL_FUNC, addr)

    def get_opaque_id(self, addr: int) -> int:
        """v1-only opaque-constant identity (NOT a v2 category).

        v1 "opaque" splits across multiple v2 categories
        (`ro_data_ptr`/`rw_data_ptr`/`string_ptr`/`valued_const`), so this
        keeps its own counter/dict until callers migrate to the appropriate
        `get_identity(Category.*, ...)` call.
        """
        if addr and addr in self.opaque_ids:
            return self.opaque_ids[addr]

        opaque_id = self.opaque_counter
        if addr:
            self.opaque_ids[addr] = opaque_id
        self.opaque_counter += 1
        return opaque_id

    def reset(self):
        """v1 reset: per-category state + v1-only opaque state."""
        self.reset_function()
        self.opaque_counter = 0
        self.opaque_ids.clear()


class LitTokenType(Enum):
    """Enum to specify if a token is a regular token, Lit_Start, or Lit_End token"""

    REGULAR = "regular"
    LIT_START = "lit_start"
    LIT_END = "lit_end"
