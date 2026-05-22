"""Token-string prefix grammar for variant-axis vocab entries.

Single concern: convert a ``BinaryVersionInfo`` into the canonical
ordered list of prefixed token strings the unified vocab will hold.
This module knows the prefix conventions; it does not touch a
``VocabularyManager`` or any file handle.

Prefix table (matches the plan ``memoized-booping-wren.md`` § "Token-
string prefixes"):

    arch              -> ``arch:<alias>``  (alias from arch_to_variant_arch)
    compiler          -> ``comp:<compiler>``
    compiler version  -> ``cver:<compiler>:<version>``
    opt level         -> ``opt:<opt>``     (leading dash stripped)
    metadata k/v      -> ``<key>:<value>`` (one token per (key, value);
                        keys are guaranteed by ``inventory.add()`` to
                        contain no ``:`` so the decoder can split on the
                        first colon to recover ``(key, value)``)

``cver`` is namespaced under the compiler family because version
strings collide across compilers (``cver:gcc:13.2.0`` !=
``cver:clang:13.2.0``).

The two helpers split positional axes from metadata so the encoder can
lay out the binary record (positional first, then metadata).
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Tuple

from tokenizer.arch_translation import arch_to_variant_arch


# Prefix string constants — exported so other modules (encoder
# decoder, tests, future renderers) can reference them without
# duplicating the literal.
ARCH_PREFIX = "arch:"
COMP_PREFIX = "comp:"
CVER_PREFIX = "cver:"
OPT_PREFIX = "opt:"

# Number of positional axes encoded at the head of every variant
# record. Co-versioned with the on-disk bin layout — see plan §
# "Co-versioning between unified vocab and per-record bin layout".
N_POSITIONAL_AXES = 4


def _strip_opt_dash(opt: str) -> str:
    """``-O2`` -> ``O2``; identity if no leading dash."""
    return opt[1:] if opt.startswith("-") else opt


def build_arch_token(arch: str) -> str:
    return f"{ARCH_PREFIX}{arch_to_variant_arch(arch)}"


def build_comp_token(compiler: str) -> str:
    return f"{COMP_PREFIX}{compiler}"


def build_cver_token(compiler: str, version: str) -> str:
    return f"{CVER_PREFIX}{compiler}:{version}"


def build_opt_token(opt: str) -> str:
    return f"{OPT_PREFIX}{_strip_opt_dash(opt)}"


def build_metadata_tokens(
    extra_metadata: Mapping[str, Any],
) -> List[Tuple[str, List[str]]]:
    """Group metadata into ``[(key, [token_str, ...]), ...]``.

    Keys are emitted in alphabetical order; within a key, multi-valued
    entries are coerced via ``str()`` then sorted alphabetically. The
    explicit ``str()`` coercion mirrors the ``encode_flags`` precedent
    at ``tokenizer/memmap_builder/variants.py:64`` and protects against
    ``TypeError`` on mixed-type value lists (e.g. ``[1, "a", True]``).

    A scalar value is treated as a length-1 list — schema regularity
    matches the always-list decoder shape called out in the plan.
    """
    out: List[Tuple[str, List[str]]] = []
    for key in sorted(extra_metadata):
        raw = extra_metadata[key]
        # Treat str/bytes as scalars (else iter would split chars).
        if isinstance(raw, (list, tuple, set, frozenset)):
            values_iter: Iterable[Any] = raw
        else:
            values_iter = (raw,)
        sorted_values = sorted(str(v) for v in values_iter)
        out.append((key, [f"{key}:{value}" for value in sorted_values]))
    return out


def build_axis_strings(version_info: Any) -> List[str]:
    """Return all prefixed token strings for one variant.

    Layout: ``[arch, comp, cver, opt, *metadata]`` where the metadata
    tail is keys sorted alphabetically and values within each key
    sorted via ``sorted(str(v) for v in values)``.

    ``version_info`` is duck-typed against ``BinaryVersionInfo`` /
    ``VariantInfo`` — needs ``arch``, ``compiler``, ``opt``,
    ``extra_metadata`` plus one of ``compilerversion`` (builder shape)
    or ``compiler_version`` (variant_info shape).
    """
    version = getattr(
        version_info,
        "compilerversion",
        getattr(version_info, "compiler_version", None),
    )
    if version is None:
        raise AttributeError(
            "version_info missing compilerversion / compiler_version"
        )
    positional = [
        build_arch_token(version_info.arch),
        build_comp_token(version_info.compiler),
        build_cver_token(version_info.compiler, version),
        build_opt_token(version_info.opt),
    ]
    metadata_tokens: List[str] = []
    for _key, values in build_metadata_tokens(version_info.extra_metadata):
        metadata_tokens.extend(values)
    return positional + metadata_tokens
