"""Assert ``TokenizerTask`` emits ``TaskInfo.task_id`` matching the
canonical ``TokenizerIdentifier.identifier_key()`` string.

The framework's memprofile sampler is gated on a non-None ``task_id``
(it uses the string as the per-task output filename key). Without this
field populated the sampler silently short-circuits with a debug log,
so this contract is what unlocks the feature on our side.
"""

from __future__ import annotations

from pathlib import Path

from dynrunner.tokenize.identifier import TokenizerIdentifier
from dynrunner.tokenize.tokenizer_task import TokenizerTask
from tokenizer.binary_discovery import BinaryHandle
from tokenizer.variant_info import VariantInfo


def _make_pair(
    source_root: Path,
    *,
    pkg: str = "minigzip",
    arch: str = "x64",
    compiler: str = "gcc",
    compiler_version: str = "7",
    opt: str = "Os",
    size: int = 4096,
) -> tuple[BinaryHandle, VariantInfo, int]:
    """Synthesise one ``(handle, variant, size)`` triple for the sort
    helper. ``BinaryHandle.path`` is a real file under ``source_root``
    so ``relative_to(source_root)`` works at TaskInfo construction.
    """
    filename = f"{arch}-{compiler}-{compiler_version}-{opt}_{pkg}"
    path = source_root / filename
    path.write_bytes(b"")
    variant = VariantInfo(
        arch=arch,
        compiler=compiler,
        compiler_version=compiler_version,
        opt=opt,
        pkg=pkg,
    )
    return BinaryHandle(path=path), variant, size


def test_task_id_matches_identifier_key(tmp_path: Path) -> None:
    pair = _make_pair(tmp_path)
    handle, variant, _size = pair

    expected = TokenizerIdentifier(
        binary_name=variant.pkg,
        platform=variant.arch,
        compiler=variant.compiler,
        version=variant.compiler_version,
        opt_level=variant.opt,
    ).identifier_key()

    [task_info] = list(TokenizerTask._sort_and_tag_pairs([pair], tmp_path))

    assert task_info.task_id == expected
    assert expected == "minigzip/x64/gcc/7/Os"


def test_task_id_distinct_per_variant(tmp_path: Path) -> None:
    pairs = [
        _make_pair(tmp_path, pkg="minigzip", arch="x64"),
        _make_pair(tmp_path, pkg="adler32_combine", arch="arm32"),
    ]
    emitted = list(TokenizerTask._sort_and_tag_pairs(pairs, tmp_path))
    task_ids = [ti.task_id for ti in emitted]
    assert len(set(task_ids)) == len(task_ids), task_ids
    assert all(tid is not None for tid in task_ids)
