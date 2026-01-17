#!/usr/bin/env python3
"""
Generate a detailed `_structure_detailed.md` with function signatures,
class fields (class-level and inferred instance attributes) and methods.

The written markdown places the entire tree inside a fenced code block.

Usage:
    python .generate_structure_detailed.py

Notes:
 - Tries to include annotations and default expressions using ast.unparse when available.
 - Infers instance attributes by scanning `self.<name>` assignments inside `__init__`.
 - Skips common excluded directories.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "_structure_detailed.md"

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
}


# ---- helpers for formatting AST nodes to readable snippets ----
def ast_to_str(node: Optional[ast.AST]) -> str:
    """Return a short string representation of an AST node (annotation or default)."""
    if node is None:
        return ""
    # Prefer ast.unparse when available (py3.9+)
    try:
        return ast.unparse(node)
    except Exception:
        # best-effort fallbacks
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Attribute):
            val = ast_to_str(node.value)
            return f"{val}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return f"{ast_to_str(node.value)}[{ast_to_str(node.slice)}]"
        if isinstance(node, ast.Tuple):
            return "(" + ", ".join(ast_to_str(elt) for elt in node.elts) + ")"
        return "<expr>"


def format_arg(arg: ast.arg, default: Optional[ast.AST] = None) -> str:
    """Format a single argument with optional default and annotation."""
    name = arg.arg
    ann = ast_to_str(arg.annotation) if getattr(arg, "annotation", None) is not None else ""
    piece = name
    if ann:
        piece += f": {ann}"
    if default is not None:
        piece += f" = {ast_to_str(default)}"
    return piece


def format_signature(args: ast.arguments, returns: Optional[ast.AST]) -> str:
    """Create a readable function signature from ast.arguments and return annotation."""
    parts: List[str] = []

    # positional-only (py3.8+)
    posonly = getattr(args, "posonlyargs", []) or []
    all_pos = list(posonly) + list(args.args)

    # defaults align to last N of all_pos
    defaults = list(args.defaults)
    num_defaults = len(defaults)
    default_map: Dict[int, ast.AST] = {}
    if num_defaults:
        for i, d in enumerate(defaults, start=len(all_pos) - num_defaults):
            default_map[i] = d

    for i, a in enumerate(all_pos):
        default = default_map.get(i)
        parts.append(format_arg(a, default))

    if posonly:
        # indicate positional-only with '/'
        parts.append("/")

    # vararg
    if args.vararg:
        var = args.vararg
        ann = ast_to_str(var.annotation) if getattr(var, "annotation", None) is not None else ""
        s = f"*{var.arg}"
        if ann:
            s += f": {ann}"
        parts.append(s)
    elif args.kwonlyargs:
        # If there are kw-only args but no vararg, need bare '*' separator
        parts.append("*")

    # kwonlyargs and their defaults (kw_defaults align 1-to-1)
    for a, d in zip(list(args.kwonlyargs), list(args.kw_defaults)):
        default = d if d is not None else None
        parts.append(format_arg(a, default))

    # kwargs
    if args.kwarg:
        kw = args.kwarg
        ann = ast_to_str(kw.annotation) if getattr(kw, "annotation", None) is not None else ""
        s = f"**{kw.arg}"
        if ann:
            s += f": {ann}"
        parts.append(s)

    sig = ", ".join(p for p in parts if p != "")
    ret = ast_to_str(returns) if returns is not None else ""
    if ret:
        return f"({sig}) -> {ret}"
    return f"({sig})"


# ---- dataclasses to hold discovered info ----
@dataclass
class FunctionInfo:
    name: str
    signature: str
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False


@dataclass
class ClassField:
    name: str
    annotation: Optional[str] = None
    value: Optional[str] = None
    defined_at: str = "class"  # "class" or "init"


@dataclass
class ClassInfo:
    name: str
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    class_fields: List[ClassField] = field(default_factory=list)
    instance_fields: List[ClassField] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    nested_classes: List["ClassInfo"] = field(default_factory=list)


@dataclass
class FileInfo:
    path: Path
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    parse_error: Optional[str] = None


# ---- parsing logic ----
def collect_python_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.rglob("*.py")):
        if set(p.parts) & EXCLUDE_DIRS:
            continue
        files.append(p)
    return files


def extract_instance_fields_from_init(node: ast.FunctionDef) -> List[ClassField]:
    """Scan __init__ body for assignments to self.x and return list of ClassField."""
    fields: Dict[str, ClassField] = {}
    for stmt in node.body:
        # simple assignments: self.x = ...
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    name = target.attr
                    val = ast_to_str(stmt.value) if stmt.value is not None else None
                    if name not in fields:
                        fields[name] = ClassField(name=name, value=val, defined_at="init")
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                name = target.attr
                ann = ast_to_str(stmt.annotation) if getattr(stmt, "annotation", None) is not None else None
                val = ast_to_str(stmt.value) if getattr(stmt, "value", None) is not None else None
                fields[name] = ClassField(name=name, annotation=ann, value=val, defined_at="init")
        # handle simple if blocks with assignments at top level of body (not exhaustive)
    return list(fields.values())


def parse_class(node: ast.ClassDef) -> ClassInfo:
    ci = ClassInfo(
        name=node.name,
        bases=[ast_to_str(b) for b in node.bases],
        decorators=[ast_to_str(d) for d in node.decorator_list],
    )
    for sub in node.body:
        if isinstance(sub, ast.FunctionDef) or isinstance(sub, ast.AsyncFunctionDef):
            is_async = isinstance(sub, ast.AsyncFunctionDef)
            sig = format_signature(sub.args, getattr(sub, "returns", None))
            decs = [ast_to_str(d) for d in sub.decorator_list]
            fi = FunctionInfo(name=sub.name, signature=sig, decorators=decs, is_async=is_async)
            ci.methods.append(fi)
            # if it's __init__, extract inferred instance fields
            if sub.name == "__init__":
                ci.instance_fields.extend(extract_instance_fields_from_init(sub))
        elif isinstance(sub, ast.Assign):
            # class-level assignments. Could be multiple targets.
            for targ in sub.targets:
                if isinstance(targ, ast.Name):
                    name = targ.id
                    value = ast_to_str(sub.value) if sub.value is not None else None
                    ci.class_fields.append(ClassField(name=name, value=value, defined_at="class"))
        elif isinstance(sub, ast.AnnAssign):
            # annotated class-level attribute
            target = sub.target
            if isinstance(target, ast.Name):
                name = target.id
                ann = ast_to_str(sub.annotation) if getattr(sub, "annotation", None) is not None else None
                val = ast_to_str(sub.value) if getattr(sub, "value", None) is not None else None
                ci.class_fields.append(ClassField(name=name, annotation=ann, value=val, defined_at="class"))
        elif isinstance(sub, ast.ClassDef):
            # nested class
            ci.nested_classes.append(parse_class(sub))
        # ignore others
    # deduplicate instance fields by name while preserving first seen
    seen = set()
    uniq_insts: List[ClassField] = []
    for f in ci.instance_fields:
        if f.name not in seen:
            uniq_insts.append(f)
            seen.add(f.name)
    ci.instance_fields = uniq_insts
    return ci


def parse_file(path: Path) -> FileInfo:
    info = FileInfo(path=path)
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except Exception as e:
        info.parse_error = f"{type(e).__name__}: {e}"
        return info

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            is_async = isinstance(node, ast.AsyncFunctionDef)
            sig = format_signature(node.args, getattr(node, "returns", None))
            decs = [ast_to_str(d) for d in node.decorator_list]
            fi = FunctionInfo(name=node.name, signature=sig, decorators=decs, is_async=is_async)
            info.functions.append(fi)
        elif isinstance(node, ast.ClassDef):
            info.classes.append(parse_class(node))
        # ignore imports, assigns at module level
    return info


# ---- tree building and rendering ----
def build_path_tree(files: List[Path], root: Path) -> Dict:
    tree: Dict[str, Union[dict, FileInfo]] = {}
    for f in sorted(files):
        rel = f.relative_to(root)
        parts = list(rel.parts)
        node = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                node[part] = parse_file(f)
            else:
                node = node.setdefault(part, {})
    return tree


def render_class(ci: ClassInfo, indent: str) -> List[str]:
    lines: List[str] = []
    base_str = f"({', '.join(ci.bases)})" if ci.bases else ""
    decs = (" [" + " ".join(ci.decorators) + "]") if ci.decorators else ""
    lines.append(f"{indent}class {ci.name}{base_str}{decs}")
    next_indent = indent + "    "
    # class fields
    if ci.class_fields:
        lines.append(f"{next_indent}# class attributes")
        for f in ci.class_fields:
            ann = f": {f.annotation}" if f.annotation else ""
            val = f" = {f.value}" if f.value else ""
            lines.append(f"{next_indent}{f.name}{ann}{val}")
    # instance fields
    if ci.instance_fields:
        lines.append(f"{next_indent}# instance attributes (inferred from __init__)")
        for f in ci.instance_fields:
            ann = f": {f.annotation}" if f.annotation else ""
            val = f" = {f.value}" if f.value else ""
            lines.append(f"{next_indent}{f.name}{ann}{val}")
    # methods
    if ci.methods:
        lines.append(f"{next_indent}# methods")
        for m in ci.methods:
            decs = (" [" + " ".join(m.decorators) + "]") if m.decorators else ""
            async_prefix = "async " if m.is_async else ""
            lines.append(f"{next_indent}{async_prefix}def {m.name}{m.signature}{decs}")
    # nested classes
    for nested in ci.nested_classes:
        lines.append(f"{next_indent}# nested class")
        lines.extend(render_class(nested, next_indent))
    return lines


def render_file(info: FileInfo, indent: str) -> List[str]:
    lines: List[str] = []
    lines.append(f"{indent}{info.path.name}")
    next_indent = indent + "    "
    if info.parse_error:
        lines.append(f"{next_indent}[parse-error] {info.parse_error}")
        return lines
    if not info.functions and not info.classes:
        lines.append(f"{next_indent}(no top-level functions or classes)")
        return lines
    if info.functions:
        lines.append(f"{next_indent}# functions")
        for fn in info.functions:
            decs = (" [" + " ".join(fn.decorators) + "]") if fn.decorators else ""
            async_prefix = "async " if fn.is_async else ""
            lines.append(f"{next_indent}{async_prefix}def {fn.name}{fn.signature}{decs}")
    for cls in info.classes:
        lines.append(f"{next_indent}# class")
        lines.extend(render_class(cls, next_indent))
    return lines


def render_tree(node: Dict, indent: str = "") -> List[str]:
    lines: List[str] = []
    # Render directories first (dict values), then files (FileInfo)
    dirs = [k for k, v in node.items() if isinstance(v, dict)]
    files = [k for k, v in node.items() if not isinstance(v, dict)]
    for d in sorted(dirs):
        lines.append(f"{indent}{d}/")
        child = node[d]
        lines.extend(render_tree(child, indent + "    "))
    for f in sorted(files):
        info = node[f]
        assert isinstance(info, FileInfo)
        lines.extend(render_file(info, indent))
    return lines


# ---- main entrypoint ----
def main(argv: Optional[List[str]] = None) -> int:
    project_root = ROOT
    py_files = collect_python_files(project_root)
    if not py_files:
        print("No python files found.", file=sys.stderr)
        return 1
    tree = build_path_tree(py_files, project_root)
    lines = []
    header = [
        "# Project Python structure (detailed)",
        "",
        "This file was generated by `.generate_structure_detailed.py`.",
        "It includes function signatures, class-level attributes and inferred instance attributes.",
        "",
        "The tree is placed inside a fenced code block for ease of copying.",
        "",
    ]
    lines.extend(header)
    # Open code fence
    lines.append("```")
    # root folder name
    lines.append(f"{project_root.name}/")
    lines.extend(render_tree(tree, indent="    "))
    lines.append("```")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote detailed structure to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
