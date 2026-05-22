"""Distinct-token accumulator for the vocab-unifier's pass 1.

Single concern: walk many ``BinaryVersionInfo`` records, collect every
distinct prefixed token string they imply, and iterate them in a
deterministic order so the unifier can register them at stable,
low-numbered vocab IDs (``[257, 257 + n_variants)`` per the plan — the
block starts one past the eagerly-pinned ``value_negative`` marker at
id 256).

The accumulator also enforces the metadata-key invariant ``":" not in
key`` at ``add()`` time. Without this guard, a future sidecar key like
``compiler:fortify`` would silently corrupt the decoder (which splits
each metadata token on the first ``:`` to recover ``(key, value)``).
The check is a hard ``AssertionError`` — same discipline as the
``_data.bin`` uint16-ceiling guard in ``encoder.py``.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping

from .prefixes import POSITIONAL_PREFIXES, build_axis_strings


class VariantInventory:
    """Deterministic set-builder for variant-axis token strings.

    Usage (matches the unifier's Pass-3 registration loop)::

        inv = VariantInventory()
        for v in versions:
            inv.add(v)
        for token_str in inv.iter_tokens_axis_grouped():
            unified_vm.Variant_Axis(token_str)
    """

    def __init__(self) -> None:
        self._tokens: set[str] = set()

    def add(self, version_info: Any) -> None:
        """Accumulate every axis string for one variant.

        Hard-asserts metadata keys contain no ``:`` so the decoder's
        split-on-first-colon round-trip can never decode a corrupted
        ``(key, value)`` pair. Silent corruption guard per the plan §
        "Per-record bin format" invariant.

        ``extra_metadata`` is read directly (not coerced via ``or {}``)
        — the dataclass guarantees a dict via
        ``field(default_factory=dict)``, so a ``None`` here would
        signal a constructor bypass and should fail loudly rather
        than be silently treated as empty.
        """
        extra: Mapping[str, Any] = version_info.extra_metadata
        for metadata_key in extra:
            assert ":" not in metadata_key, (
                f"metadata key {metadata_key!r} contains ':', which would "
                "collide with the encoder's split-on-first-colon decoder; "
                "rename the sidecar key upstream"
            )
        for token_string in build_axis_strings(version_info):
            self._tokens.add(token_string)

    def iter_tokens(self) -> Iterator[str]:
        """Yield every distinct token string in alphabetical order.

        Alphabetical is deterministic, human-scannable, and matches the
        order the unifier wants for stable ID assignment across runs on
        the same corpus.
        """
        return iter(sorted(self._tokens))

    def iter_tokens_axis_grouped(self) -> Iterator[str]:
        """Yield tokens grouped by axis: positional axes first in the
        declared order (``arch`` -> ``comp`` -> ``cver`` -> ``opt``),
        then sidecar-key axes in alphabetical-by-prefix order. Within
        each axis, values are alphabetical.

        Same multiset as :meth:`iter_tokens`, different deterministic
        ordering — the unified vocab wants positional axes packed at
        the head of the variant block so dataloader-side decode can
        index by canonical position without a prefix lookup.

        Partitioning is by the substring up to and including the first
        ``:`` (the prefix grammar guaranteed by ``add()``'s metadata-key
        invariant); the declared positional ordering lives in
        :data:`tokenizer.variant_tokens.prefixes.POSITIONAL_PREFIXES`
        so this iterator does not duplicate the axis sequence.
        """
        buckets: dict[str, list[str]] = {}
        for token in self._tokens:
            key, sep, _ = token.partition(":")
            assert sep == ":", (
                f"variant token {token!r} missing ':' prefix delimiter; "
                "every token added via add() goes through build_axis_strings "
                "which always emits a prefix"
            )
            buckets.setdefault(key + ":", []).append(token)

        for prefix in POSITIONAL_PREFIXES:
            if prefix in buckets:
                yield from sorted(buckets.pop(prefix))

        for prefix in sorted(buckets):
            yield from sorted(buckets[prefix])

    def __len__(self) -> int:
        return len(self._tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._tokens

    def update(self, version_infos: Iterable[Any]) -> None:
        """Convenience: add every version_info in an iterable."""
        for vi in version_infos:
            self.add(vi)
