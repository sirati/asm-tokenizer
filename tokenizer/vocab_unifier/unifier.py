import csv
from pathlib import Path

import numpy as np

from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager

from .loader import load_vocab_manager
from .saver import save_vocabulary


def unify_vocab(csv_files: list[Path], unified_vocab_file: Path) -> None:
    unified_vm = VocabularyManager(platform=None)

    for csv_file in csv_files:
        print(f"Loading vocabulary from {csv_file}")
        current_vocab_manager = load_vocab_manager(csv_file)
        mappings = np.full_like(current_vocab_manager.id_to_token_type, -1, dtype=np.int32)

        for tokens in current_vocab_manager.iter_representative_tokens():
            original = tokens.get_token_ids()
            mapped = tokens.register_on_vocab_manager(unified_vm).get_token_ids()
            assert len(original) == len(mapped)
            for original_id, mapped_id in zip(original, mapped):
                mappings[original_id] = mapped_id

        assert np.all(mappings >= 0)

        mapping_file_path = csv_file.with_suffix(".mapping.b64c")
        with open(mapping_file_path, "w", newline="", encoding="ascii") as mapping_file:
            mapping_file.write(ndarray_to_base64(mappings))

    print(f"Saving unified vocabulary to {unified_vocab_file}")
    with open(unified_vocab_file, "w", newline="", encoding="ascii") as csvfile:
        writer = csv.writer(csvfile)
        save_vocabulary(unified_vm, writer)
