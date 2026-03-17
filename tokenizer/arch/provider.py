from abc import ABC, abstractmethod
from typing import List

from tokenizer.constant_handler import ConstantHandler
from tokenizer.instruction_sets import InstructionSets
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Tokens


class ArchitectureProvider(ABC):
    """Encapsulates all architecture-specific tokenization logic."""

    @property
    @abstractmethod
    def platform_str(self) -> str:
        """Platform string used for token prefixing, e.g. 'x86', 'x64', 'arm32'."""
        ...

    @abstractmethod
    def load_instruction_sets(self) -> InstructionSets:
        """Load architecture-specific instruction classification data."""
        ...

    @abstractmethod
    def parse_instruction(
        self,
        instr_sets: InstructionSets,
        constant_handler: ConstantHandler,
        func_max_addr: int,
        func_min_addr: int,
        insn,
        lookup,
        text_end: int,
        text_start: int,
        vocab_manager: VocabularyManager,
        insn_tokens: List[Tokens],
    ) -> List[Tokens]:
        """Parse a single instruction into tokens. Appends to and returns insn_tokens."""
        ...
