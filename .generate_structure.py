#!/usr/bin/env python3
"""
Generate `_structure.md` describing python files and their inner symbols
(classes, functions, methods) in a tree-like layout.

Usage:
    python .generate_structure.py

The script scans the directory that contains this file (the package root),
walks Python files, parses their AST and writes `./_structure.md`.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".direnv",
    ".ruff_cache",
    ".zed",
    "out",
    "output_meta",
    "build",
    "dist",
    ".eggs",
}


@dataclass
class Symbol:
    """
    Represents a symbol inside a Python file: a top-level function or class,
    or a method/inner-class inside a class.
    """

    kind: str  # "class", "function", "async-function", "method", "async-method"
    name: str
    children: List["Symbol"] = field(default_factory=list)


@dataclass
class FileInfo:
    """
    Representation of a Python file's discovered symbols and parse error (if any).
    """

    path: Path
    symbols: List[Symbol] = field(default_factory=list)
    parse_error: Optional[str] = None


def iter_python_files(root: Path) -> Iterable[Path]:
    """
    Yield python files under `root`, skipping excluded directories.
    """
    for p in sorted(root.rglob("*.py")):
        # Skip files in excluded directories
        parts = set(p.parts)
        if parts & EXCLUDE_DIRS:
            continue
        yield p


def symbol_from_classdef(node: ast.ClassDef) -> Symbol:
    """
    Convert an ast.ClassDef into a Symbol, recursively including nested defs.
    """
    sym = Symbol(kind="class", name=node.name)
    for sub in node.body:
        if isinstance(sub, ast.FunctionDef):
            # method
            child = Symbol(kind="method", name=sub.name)
            # nested functions inside methods are not shown as methods for brevity,
            # but nested classes are included
            for inner in sub.body:
                if isinstance(inner, ast.ClassDef):
                    child.children.append(symbol_from_classdef(inner))
            sym.children.append(child)
        elif isinstance(sub, ast.AsyncFunctionDef):
            child = Symbol(kind="async-method", name=sub.name)
            for inner in sub.body:
                if isinstance(inner, ast.ClassDef):
                    child.children.append(symbol_from_classdef(inner))
            sym.children.append(child)
        elif isinstance(sub, ast.ClassDef):
            # nested class
            sym.children.append(symbol_from_classdef(sub))
        # other nodes (assign, import, etc.) are ignored
    return sym


def parse_file(path: Path) -> FileInfo:
    """
    Parse a python file and return FileInfo with discovered top-level symbols.
    """
    info = FileInfo(path=path)
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except Exception as e:
        info.parse_error = f"{type(e).__name__}: {e}"
        return info

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            info.symbols.append(Symbol(kind="function", name=node.name))
        elif isinstance(node, ast.AsyncFunctionDef):
            info.symbols.append(Symbol(kind="async-function", name=node.name))
        elif isinstance(node, ast.ClassDef):
            info.symbols.append(symbol_from_classdef(node))
        # ignore other node types (imports, assignments, etc.)
    return info


def build_path_tree(files: List[Path], root: Path) -> Dict[str, Union[dict, FileInfo]]:
    """
    Build nested dictionary representing directory tree.

    Keys: directory or filename.
    Values: nested dict for directory, or FileInfo for file.
    """
    tree: Dict[str, Union[dict, FileInfo]] = {}
    for f in sorted(files):
        rel = f.relative_to(root)
        parts = list(rel.parts)
        node = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # file
                node[part] = parse_file(f)
            else:
                node = node.setdefault(part, {})
    return tree


def render_symbol_lines(sym: Symbol, indent: str) -> List[str]:
    """
    Render a symbol and its children to lines with proper indentation.
    """
    lines: List[str] = []
    prefix = f"{indent}└─ " if indent else "└─ "
    kind_label = {
        "class": "class",
        "function": "def",
        "async-function": "async def",
        "method": "def",
        "async-method": "async def",
    }.get(sym.kind, sym.kind)
    lines.append(f"{prefix}{kind_label} {sym.name}")
    for child in sym.children:
        # Increase indent for children: replace '└─ ' with '   '
        child_indent = indent + "   "
        lines.extend(render_symbol_lines(child, child_indent))
    return lines


def render_file_info(info: FileInfo, indent: str) -> List[str]:
    """
    Render the file entry and its symbols.
    """
    lines: List[str] = []
    # File line
    file_prefix = f"{indent}└─ " if indent else "└─ "
    lines.append(f"{file_prefix}{info.path.name}")
    next_indent = indent + "   "
    if info.parse_error:
        lines.append(f"{next_indent}[parse-error] {info.parse_error}")
        return lines
    if not info.symbols:
        lines.append(f"{next_indent}(no top-level classes/functions)")
        return lines
    for sym in info.symbols:
        lines.extend(render_symbol_lines(sym, next_indent))
    return lines


def render_tree(node: Dict[str, Union[dict, FileInfo]], indent: str = "") -> List[str]:
    """
    Recursively render the directory/file tree into text lines.
    """
    lines: List[str] = []
    # Sort directories first, then files
    dirs = [k for k, v in node.items() if isinstance(v, dict)]
    files = [k for k, v in node.items() if not isinstance(v, dict)]
    for d in sorted(dirs):
        prefix = f"{indent}└─ " if indent else "└─ "
        lines.append(f"{prefix}{d}/")
        child = node[d]
        lines.extend(render_tree(child, indent + "   "))
    for f in sorted(files):
        info = node[f]
        assert isinstance(info, FileInfo)
        lines.extend(render_file_info(info, indent))
    return lines


def write_structure_md(root: Path, tree_lines: Iterable[str], outname: str = "_structure.md") -> Path:
    """
    Write the rendered lines to the output markdown file at `root/outname`.
    """
    outpath = root / outname
    header = [
        "# Project Python structure",
        "",
        "This file was generated by `.generate_structure.py`. It lists python files and their",
        "top-level symbols (classes, functions) and methods / nested classes.",
        "",
    ]
    content = "\n".join(header + list(tree_lines)) + "\n"
    outpath.write_text(content, encoding="utf-8")
    return outpath


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv or sys.argv[1:])
    script_path = Path(__file__).resolve()
    project_root = script_path.parent  # script is placed in asm-tokenizer root
    py_files = [p for p in iter_python_files(project_root)]

    if not py_files:
        print("No python files found.", file=sys.stderr)
        return 1

    tree = build_path_tree(py_files, project_root)
    lines = render_tree(tree, indent="")
    outpath = write_structure_md(project_root, lines)
    print(f"Wrote structure to {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
