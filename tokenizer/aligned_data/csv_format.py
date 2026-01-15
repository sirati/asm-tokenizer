from typing import List, Tuple


def format_inlining_dict(inlining_list: List) -> str:
    """Format inlining data as semicolon-separated: idx,hex_offset,hex_length,is_matched;..."""
    if not inlining_list:
        return ""
    parts = []
    for idx, start, length, is_matched in inlining_list:
        hex_start = f"{start:x}"
        hex_length = f"{length:x}"
        parts.append(f"{idx},{hex_start},{hex_length},{is_matched}")
    return ";".join(parts)


def format_compiler_sets(compiler_sets: List[Tuple[str, str, str, str]]) -> str:
    """Format list of compiler set tuples using semicolon separation: arch,compiler,version,opt;..."""
    if not compiler_sets:
        return ""
    parts = []
    for arch, compiler, compilerversion, opt in compiler_sets:
        parts.append(f"{arch},{compiler},{compilerversion},{opt}")
    return ";".join(parts)


def format_unique_called(unique_called: List[str]) -> str:
    """Format list of function names, comma-separated with escaped commas"""
    escaped = [name.replace(",", "\\,") for name in unique_called]
    return ",".join(escaped)


def parse_escaped_function_names(called_str: str) -> List[str]:
    """Parse comma-separated function names, handling escaped commas."""
    if not called_str:
        return []
    parts = []
    current = []
    i = 0
    while i < len(called_str):
        if (
            called_str[i] == "\\"
            and i + 1 < len(called_str)
            and called_str[i + 1] == ","
        ):
            current.append(",")
            i += 2
        elif called_str[i] == ",":
            parts.append("".join(current))
            current = []
            i += 1
        else:
            current.append(called_str[i])
            i += 1
    if current:
        parts.append("".join(current))
    return parts


def parse_inlining_data(inlining_str: str) -> List[Tuple]:
    """Parse semicolon-separated inlining data: idx,hex_offset,hex_length,is_matched;..."""
    if not inlining_str:
        return []
    entries = []
    for part in inlining_str.split(";"):
        if not part:
            continue
        fields = part.split(",")
        if len(fields) >= 4:
            idx = fields[0]
            offset = int(fields[1], 16)
            length = int(fields[2], 16)
            is_matched = int(fields[3])
            entries.append((idx, offset, length, is_matched))
    return entries


def parse_compiler_sets(compiler_sets_str: str) -> List[Tuple[str, str, str, str]]:
    """Parse semicolon-separated compiler sets: arch,compiler,version,opt;..."""
    if not compiler_sets_str:
        return []
    sets = []
    for part in compiler_sets_str.split(";"):
        if not part:
            continue
        fields = part.split(",")
        if len(fields) >= 4:
            sets.append((fields[0], fields[1], fields[2], fields[3]))
    return sets
