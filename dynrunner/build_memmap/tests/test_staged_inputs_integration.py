"""The build_memmap worker reads per-binary inputs from node-local
scratch, never directly off the NFS source paths.

In container mode ``_process_payload`` must stage every surviving
version's csv + mapping (+ optional `_meta.json`) via ``staged_inputs``
and hand ``build_memmap_files`` / ``VariantInfo.from_csv`` the LOCAL
copies — so no content read hits the shared filesystem. The skip-missing
resilience and the "fail only if no survivor" contract must be preserved
across the restructure. Standalone mode stays a pass-through (originals).

``build_memmap_files`` and ``VariantInfo.from_csv`` are stubbed so the
test pins the path-routing contract without invoking the full builder;
deployment mode + the tmpfs root are forced via the staging module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tokenizer.output_staging as staging
import dynrunner.build_memmap.worker as worker
from tokenizer.memmap_builder.builder import BinaryVersionInfo


class _FakeTask:
    """Minimal Task stand-in: staged_publish in standalone/container mode
    only needs ``publish_all``; the container branch calls it with the
    published files. We record nothing — the build stub writes no files."""

    def publish_all(self, *paths: Path) -> None:  # pragma: no cover - unused
        pass


@pytest.fixture
def out_tmp(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "out-tmp"
    root.mkdir()
    monkeypatch.setattr(staging, "_SLURM_OUT_TMP", root)
    monkeypatch.setattr(staging, "is_container_deployment", lambda: True)
    return root


def _layout(base: Path) -> tuple[Path, Path, Path]:
    src = base / "src"
    vocab = base / "vocab"
    out = base / "out"
    csv = src / "binA" / "x86-gcc-9-O2.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text("csv-content")
    mapping = vocab / "binA" / "x86-gcc-9-O2.mapping.b64c"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_text("mapping-content")
    return src, vocab, out


def _payload(*entries: dict) -> str:
    import json

    return json.dumps({"versions": list(entries)})


def test_container_routes_locals_into_builder(
    tmp_path: Path, out_tmp: Path, monkeypatch
) -> None:
    src, vocab, out = _layout(tmp_path)
    scratch = out_tmp / "staged-inputs" / "build_memmap" / "binA"

    seen: dict = {}

    def fake_build(versions, output_dir, binary_name, unified_vocab_path):
        v = versions[0]
        seen["path"] = v.path
        seen["mapping_path"] = v.mapping_path
        # The builder must receive LOCAL copies, not the NFS sources.
        assert scratch in v.path.parents
        assert scratch in v.mapping_path.parents
        assert v.path.read_text() == "csv-content"
        assert v.mapping_path.read_text() == "mapping-content"

    monkeypatch.setattr(worker, "build_memmap_files", fake_build)

    worker._process_payload(
        _FakeTask(),
        "binA",
        _payload(
            {
                "csv_path": "binA/x86-gcc-9-O2.csv",
                "mapping_path": "binA/x86-gcc-9-O2.mapping.b64c",
                "arch": "x86",
                "compiler": "gcc",
                "compilerversion": "9",
                "opt": "O2",
            }
        ),
        src,
        vocab,
        out,
        tmp_path / "unified_vocab.csv",
    )

    assert "path" in seen
    # Scratch is cleaned on clean exit.
    assert not scratch.exists()


def test_container_stages_meta_for_variant_info(
    tmp_path: Path, out_tmp: Path, monkeypatch
) -> None:
    src, vocab, out = _layout(tmp_path)
    meta = src / "binA" / "_meta.json"
    meta.write_text("{}")
    scratch = out_tmp / "staged-inputs" / "build_memmap" / "binA"

    seen: dict = {}

    def fake_from_csv(csv_path, meta_path):
        # from_csv must read the LOCAL csv + meta copies.
        assert scratch in Path(csv_path).parents
        assert scratch in Path(meta_path).parents
        seen["csv"] = Path(csv_path)
        seen["meta"] = Path(meta_path)

        class _Info:
            pkg = "mypkg"
            extra_metadata = {"hardening": "full"}

        return _Info()

    captured: dict = {}

    def fake_build(versions, output_dir, binary_name, unified_vocab_path):
        captured["pkg"] = versions[0].pkg
        captured["extra"] = versions[0].extra_metadata

    monkeypatch.setattr(worker.VariantInfo, "from_csv", staticmethod(fake_from_csv))
    monkeypatch.setattr(worker, "build_memmap_files", fake_build)

    worker._process_payload(
        _FakeTask(),
        "binA",
        _payload(
            {
                "csv_path": "binA/x86-gcc-9-O2.csv",
                "mapping_path": "binA/x86-gcc-9-O2.mapping.b64c",
                "meta_path": "binA/_meta.json",
                "arch": "x86",
                "compiler": "gcc",
                "compilerversion": "9",
                "opt": "O2",
            }
        ),
        src,
        vocab,
        out,
        tmp_path / "unified_vocab.csv",
    )

    assert seen["csv"].exists() is False  # cleaned after build
    assert captured == {"pkg": "mypkg", "extra": {"hardening": "full"}}


def test_skip_missing_preserved_under_staging(
    tmp_path: Path, out_tmp: Path, monkeypatch
) -> None:
    src, vocab, out = _layout(tmp_path)

    built: dict = {}

    def fake_build(versions, output_dir, binary_name, unified_vocab_path):
        # Only the present version survives; the missing one is skipped.
        built["count"] = len(versions)
        built["arch"] = versions[0].arch

    monkeypatch.setattr(worker, "build_memmap_files", fake_build)

    worker._process_payload(
        _FakeTask(),
        "binA",
        _payload(
            {
                "csv_path": "binA/x86-gcc-9-O2.csv",
                "mapping_path": "binA/x86-gcc-9-O2.mapping.b64c",
                "arch": "x86",
                "compiler": "gcc",
                "compilerversion": "9",
                "opt": "O2",
            },
            {
                "csv_path": "binA/MISSING.csv",
                "mapping_path": "binA/MISSING.mapping.b64c",
                "arch": "arm",
                "compiler": "gcc",
                "compilerversion": "9",
                "opt": "O2",
            },
        ),
        src,
        vocab,
        out,
        tmp_path / "unified_vocab.csv",
    )

    assert built == {"count": 1, "arch": "x86"}


def test_all_missing_raises_filenotfound(
    tmp_path: Path, out_tmp: Path, monkeypatch
) -> None:
    src, vocab, out = _layout(tmp_path)
    monkeypatch.setattr(
        worker, "build_memmap_files", lambda *a, **k: None
    )

    with pytest.raises(FileNotFoundError):
        worker._process_payload(
            _FakeTask(),
            "binA",
            _payload(
                {
                    "csv_path": "binA/GONE.csv",
                    "mapping_path": "binA/GONE.mapping.b64c",
                    "arch": "x86",
                    "compiler": "gcc",
                    "compilerversion": "9",
                    "opt": "O2",
                }
            ),
            src,
            vocab,
            out,
            tmp_path / "unified_vocab.csv",
        )


def test_standalone_passes_through_nfs_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(staging, "is_container_deployment", lambda: False)
    src, vocab, out = _layout(tmp_path)
    csv = src / "binA" / "x86-gcc-9-O2.csv"

    seen: dict = {}

    def fake_build(versions, output_dir, binary_name, unified_vocab_path):
        seen["path"] = versions[0].path

    monkeypatch.setattr(worker, "build_memmap_files", fake_build)

    worker._process_payload(
        _FakeTask(),
        "binA",
        _payload(
            {
                "csv_path": "binA/x86-gcc-9-O2.csv",
                "mapping_path": "binA/x86-gcc-9-O2.mapping.b64c",
                "arch": "x86",
                "compiler": "gcc",
                "compilerversion": "9",
                "opt": "O2",
            }
        ),
        src,
        vocab,
        out,
        tmp_path / "unified_vocab.csv",
    )

    # Standalone: the builder reads the original NFS source path directly.
    assert seen["path"] == csv
