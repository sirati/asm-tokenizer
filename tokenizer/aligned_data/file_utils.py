from pathlib import Path
from typing import BinaryIO, TextIO


class FileManager:
    """Manages multiple related files for aligned data export."""

    def __init__(self, base_path: str, prefix: str):
        self.base_path = Path(base_path)
        self.prefix = prefix
        self.sections_path = self.base_path / f"{prefix}_sections.csv"
        self.data_path = self.base_path / f"{prefix}_data.bin"
        self.index_path = self.base_path / f"{prefix}_index.bin"

    def open_sections_write(self, encoding: str = "ascii") -> TextIO:
        """Open sections CSV file for writing."""
        return open(self.sections_path, "w", newline="", encoding=encoding)

    def open_data_write(self) -> BinaryIO:
        """Open data binary file for writing."""
        return open(self.data_path, "wb")

    def open_index_write(self) -> BinaryIO:
        """Open index binary file for writing."""
        return open(self.index_path, "wb")

    def open_sections_read(self, encoding: str = "ascii") -> TextIO:
        """Open sections CSV file for reading."""
        return open(self.sections_path, "r", newline="", encoding=encoding)

    def open_data_read(self) -> BinaryIO:
        """Open data binary file for reading."""
        return open(self.data_path, "rb")

    def open_index_read(self) -> BinaryIO:
        """Open index binary file for reading."""
        return open(self.index_path, "rb")


def open_warn_log(base_path: str, binary_name: str, encoding: str = "ascii") -> TextIO:
    """Open warn log file for writing."""
    warn_log_path = Path(base_path) / f"{binary_name}.warn.log"
    return open(warn_log_path, "w", encoding=encoding)


def close_files(*files):
    """Close multiple file handles."""
    for f in files:
        if f is not None:
            f.close()
