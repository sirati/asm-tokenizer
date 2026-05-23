"""End-to-end test for the :mod:`tools.run_batch_smoke` CLI driver.

Builds a synthetic single-binary memmap via the loader test fixture,
stages a v1 ``unified_vocab.csv`` alongside it, and invokes the
driver's ``main()`` programmatically. Asserts the output JSON exists
and carries the documented schema keys.

The shared fixture (``build_synthetic_binary`` + ``stage_v1_unified_vocab``)
is the same one the BinarySession lifecycle tests use; depending on
it keeps this driver-level smoke aligned with the production session
contract rather than maintaining a parallel synthetic-corpus
builder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set

from tokenizer.aligned_data.loader.tests._loader_test_support import (
    stage_v1_unified_vocab,
)
from tokenizer.aligned_data.loader.tests._session_fixture import (
    build_synthetic_binary,
)

from tools.run_batch_smoke import main


_EXPECTED_TOP_LEVEL_KEYS: Set[str] = {
    "schema_version",
    "timestamp",
    "tip",
    "memmap_dir",
    "config",
    "per_binary",
    "aggregate",
    "wall_seconds",
}

_EXPECTED_CONFIG_KEYS: Set[str] = {
    "num_variants_per_section",
    "context_len",
    "max_depth",
    "variant_padding",
    "max_functions_per_binary",
    "seed",
}

_EXPECTED_PER_BINARY_KEYS: Set[str] = {
    "batch_size",
    "tokens_shape",
    "total_identity_chunks",
    "total_number_chunks",
    "section_count",
    "wall_seconds",
}

_EXPECTED_AGGREGATE_KEYS: Set[str] = {
    "batch_size",
    "total_tokens",
    "total_identity_chunks",
    "total_number_chunks",
    "section_count",
}


def test_main_writes_expected_schema(tmp_path: Path) -> None:
    """``main([...])`` runs end-to-end against a synthetic corpus and
    writes a JSON file with the documented top-level + nested schema.

    The synthetic fixture builds one matched section (2 variants)
    + one unmatched section. The driver only walks the matched arm so
    ``per_binary[<bin>].section_count`` is exactly 1 and the aggregate
    is a single-binary roll-up of that block.
    """
    fb = build_synthetic_binary(tmp_path)
    stage_v1_unified_vocab(tmp_path)
    binary_name = fb["binary_name"]

    output_path = tmp_path / "batch_smoke_results.json"

    rc = main(
        [
            "--memmap-dir",
            str(tmp_path),
            "--output",
            str(output_path),
            "--max-functions-per-binary",
            "4",
            "--num-variants-per-section",
            "2",
            "--context-len",
            "32",
            "--max-depth",
            "1",
            "--seed",
            "7",
            "--variant-padding",
            "pad_null",
        ]
    )
    assert rc == 0
    assert output_path.is_file()

    payload = json.loads(output_path.read_text())

    # Top-level schema
    assert set(payload.keys()) == _EXPECTED_TOP_LEVEL_KEYS
    assert payload["schema_version"] == 1
    assert payload["memmap_dir"] == str(tmp_path.resolve())

    # Config block carries every CLI knob the driver exposes
    assert set(payload["config"].keys()) == _EXPECTED_CONFIG_KEYS
    assert payload["config"]["num_variants_per_section"] == 2
    assert payload["config"]["context_len"] == 32
    assert payload["config"]["max_depth"] == 1
    assert payload["config"]["variant_padding"] == "pad_null"
    assert payload["config"]["max_functions_per_binary"] == 4
    assert payload["config"]["seed"] == 7

    # Per-binary block exists for our synthetic binary
    per_binary = payload["per_binary"]
    assert binary_name in per_binary
    block = per_binary[binary_name]
    assert set(block.keys()) == _EXPECTED_PER_BINARY_KEYS
    assert block["section_count"] == 1
    # 1 section x 2 variants = 2 rows; columns = context_len.
    assert block["tokens_shape"] == [2, 32]
    assert block["batch_size"] == 2

    # Aggregate block mirrors the per-binary roll-up (single binary).
    aggregate_block = payload["aggregate"]
    assert set(aggregate_block.keys()) == _EXPECTED_AGGREGATE_KEYS
    assert aggregate_block["batch_size"] == block["batch_size"]
    assert aggregate_block["section_count"] == block["section_count"]
    assert (
        aggregate_block["total_tokens"]
        == block["tokens_shape"][0] * block["tokens_shape"][1]
    )
