import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FileInfo:
    relative_path: str
    size: int
    mtime: float
    file_hash: str


@dataclass
class DirectoryHash:
    overall_hash: str
    files: dict[str, FileInfo]


HASH_SIZE = 32
ZERO_HASH = b"\x00" * HASH_SIZE

_cached_tokenizer_hash: str | None = None
_tokenizer_dir: Path = Path(__file__).parent.resolve()
_hash_cache_path: Path = _tokenizer_dir / ".tokenizer_hash_cache.json"


def _compute_file_hash(file_path: Path) -> str:
    hasher = hashlib.md5()
    hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _load_hash_cache(cache_path: Path) -> DirectoryHash | None:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        files = {
            rel_path: FileInfo(
                relative_path=rel_path,
                size=info["size"],
                mtime=info["mtime"],
                file_hash=info["file_hash"],
            )
            for rel_path, info in data["files"].items()
        }
        return DirectoryHash(overall_hash=data["overall_hash"], files=files)
    except Exception:
        return None


def _save_hash_cache(cache_path: Path, dir_hash: DirectoryHash) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "overall_hash": dir_hash.overall_hash,
        "files": {
            info.relative_path: {
                "size": info.size,
                "mtime": info.mtime,
                "file_hash": info.file_hash,
            }
            for info in dir_hash.files.values()
        },
    }
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


def compute_directory_hash(directory: Path, cache_path: Path) -> str:
    cached = _load_hash_cache(cache_path)
    cached_files = cached.files if cached else {}

    all_files = sorted(list(directory.rglob("*.py")) + list(directory.rglob("*.json")))
    all_files = [f for f in all_files if not any(part.startswith(".") for part in f.relative_to(directory).parts)]
    current_files: dict[str, FileInfo] = {}

    for py_file in all_files:
        rel_path = str(py_file.relative_to(directory))
        stat = py_file.stat()
        size = stat.st_size
        mtime = stat.st_mtime

        cached_info = cached_files.get(rel_path)
        if cached_info and cached_info.size == size and cached_info.mtime == mtime:
            file_hash = cached_info.file_hash
        else:
            file_hash = _compute_file_hash(py_file)

        current_files[rel_path] = FileInfo(
            relative_path=rel_path,
            size=size,
            mtime=mtime,
            file_hash=file_hash,
        )

    hasher = hashlib.md5()
    for rel_path in sorted(current_files.keys()):
        file_info = current_files[rel_path]
        hasher.update(rel_path.encode())
        hasher.update(file_info.file_hash.encode())

    overall_hash = hasher.hexdigest()
    dir_hash = DirectoryHash(overall_hash=overall_hash, files=current_files)
    _save_hash_cache(cache_path, dir_hash)

    return overall_hash


def save_pickle_with_hash(file_path: Path, data: Any, expected_hash: str) -> None:
    hash_bytes = expected_hash.encode("ascii")
    if len(hash_bytes) != HASH_SIZE:
        raise ValueError(f"Hash must be exactly {HASH_SIZE} bytes when encoded as ASCII")

    with open(file_path, "wb") as f:
        f.write(ZERO_HASH)
        pickle.dump(data, f)
        f.seek(0)
        f.write(hash_bytes)


def load_pickle_with_hash(file_path: Path, expected_hash: str) -> Any:
    with open(file_path, "rb") as f:
        stored_hash_bytes = f.read(HASH_SIZE)

    if stored_hash_bytes == ZERO_HASH:
        raise ValueError("Pickle file has zero hash (incomplete write detected)")

    stored_hash = stored_hash_bytes.decode("ascii")
    if stored_hash != expected_hash:
        raise ValueError(
            f"Hash mismatch: expected {expected_hash}, got {stored_hash} (tokenizer code changed or pickle corrupted)"
        )

    with open(file_path, "rb") as f:
        f.seek(HASH_SIZE)
        return pickle.load(f)


def get_pickle_hash(file_path: Path) -> str | None:
    if not file_path.exists():
        return None
    try:
        with open(file_path, "rb") as f:
            hash_bytes = f.read(HASH_SIZE)
        if hash_bytes == ZERO_HASH:
            return None
        return hash_bytes.decode("ascii")
    except Exception:
        return None


def get_current_hash() -> str:
    global _cached_tokenizer_hash
    if _cached_tokenizer_hash is None:
        _cached_tokenizer_hash = compute_directory_hash(_tokenizer_dir, _hash_cache_path)
    return _cached_tokenizer_hash


def has_valid_pickle(file_path: Path) -> bool:
    if not file_path.exists():
        return False

    stored_hash = get_pickle_hash(file_path)
    if stored_hash is None:
        file_path.unlink(missing_ok=True)
        return False

    try:
        current_hash = get_current_hash()
        if stored_hash != current_hash:
            file_path.unlink(missing_ok=True)
            return False
    except Exception:
        file_path.unlink(missing_ok=True)
        return False

    return True


def try_load_pickle(file_path: Path, logger: logging.Logger | None = None) -> Any | None:
    if not has_valid_pickle(file_path):
        return None

    try:
        current_hash = get_current_hash()
        data = load_pickle_with_hash(file_path, current_hash)
        return data
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load pickle from {file_path}: {e}")
            logger.info(f"Deleting corrupted pickle file: {file_path}")
        file_path.unlink(missing_ok=True)
        return None


def save_pickle(file_path: Path, data: Any) -> None:
    current_hash = get_current_hash()
    save_pickle_with_hash(file_path, data, current_hash)
