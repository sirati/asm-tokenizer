"""Unit tests for the pure name-substring matchers in :mod:`._search_match`.

The pure helpers (:func:`first_function_match`, :func:`iter_function_matches`,
:func:`next_match_index`) are independent of textual so these tests run in
the default ``nix develop`` shell. Higher-level integration coverage of
the search bar lives in :mod:`test_app_search_bar` (textual-gated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.inspector._app._search_match import (
    first_function_match,
    iter_function_matches,
    next_match_index,
)
from tokenizer.inspector._render._protocol import FunctionHandle
from tokenizer.inspector._tree_model import FunctionNode


@dataclass
class _FakeFactory:
    """Stand-in factory the helpers never call into (no ``.make()`` reached)."""


def _function_node(name: str) -> FunctionNode:
    handle = FunctionHandle(arm=SectionKind.MATCHED, idx=0, name=name)
    return FunctionNode(factory=_FakeFactory(), handle=handle)  # type: ignore[arg-type]


class _FakeTreeNode:
    """Minimal stand-in for :class:`textual.widgets._tree.TreeNode`.

    The pure helpers only read ``.data`` (a :class:`FunctionNode`) and
    ``._line`` (an int). Carrying just those two fields lets the tests
    skip the textual stack.
    """

    def __init__(self, model: object, line: int = 0) -> None:
        self.data = model
        self._line = line


def _make_rows(names: list[str]) -> list[_FakeTreeNode]:
    return [_FakeTreeNode(_function_node(n), line=i) for i, n in enumerate(names)]


# ---------------------------------------------------------------------------
# first_function_match / iter_function_matches
# ---------------------------------------------------------------------------


def test_first_function_match_returns_first_hit_in_declared_order():
    rows = _make_rows(["calloc", "malloc", "free"])
    match = first_function_match(rows, "loc")
    assert match is rows[0]  # "calloc" matched before "malloc"


def test_first_function_match_skips_non_function_nodes():
    rows = _make_rows(["calloc"])
    rows.insert(0, _FakeTreeNode(object(), line=99))  # non-FunctionNode payload
    match = first_function_match(rows, "calloc")
    assert match is rows[1]


def test_first_function_match_returns_none_on_miss():
    rows = _make_rows(["calloc", "malloc"])
    assert first_function_match(rows, "zzz") is None


def test_iter_function_matches_yields_every_hit_in_order():
    rows = _make_rows(["calloc", "malloc", "free", "callee"])
    results = list(iter_function_matches(rows, "cal"))
    assert results == [rows[0], rows[3]]


# ---------------------------------------------------------------------------
# next_match_index
# ---------------------------------------------------------------------------


def test_next_match_index_cursor_on_match_wraps_forward():
    rows = _make_rows(["calloc", "malloc"])
    matches = list(iter_function_matches(rows, "loc"))
    assert len(matches) == 2
    # Cursor on first match -> next = second.
    assert next_match_index(matches, matches[0], matches[0]._line, forward=True) == 1
    # Cursor on second match -> wraps to first.
    assert next_match_index(matches, matches[1], matches[1]._line, forward=True) == 0


def test_next_match_index_cursor_on_match_wraps_backward():
    rows = _make_rows(["calloc", "malloc"])
    matches = list(iter_function_matches(rows, "loc"))
    # Cursor on second match -> previous = first.
    assert next_match_index(matches, matches[1], matches[1]._line, forward=False) == 0
    # Cursor on first match -> wraps to last (second).
    assert next_match_index(matches, matches[0], matches[0]._line, forward=False) == 1


def test_next_match_index_cursor_on_non_match_forward():
    rows = _make_rows(["aaa", "bbb_match", "ccc", "ddd_match", "eee"])
    matches = list(iter_function_matches(rows, "match"))
    # Cursor sits on row "ccc" (line 2): forward should land on the
    # first match AFTER line 2 -> "ddd_match" (line 3).
    cursor = rows[2]
    assert next_match_index(matches, cursor, cursor._line, forward=True) == 1


def test_next_match_index_cursor_on_non_match_backward():
    rows = _make_rows(["aaa", "bbb_match", "ccc", "ddd_match", "eee"])
    matches = list(iter_function_matches(rows, "match"))
    # Cursor on "ccc" (line 2): backward -> first match BEFORE line 2
    # is "bbb_match" (line 1).
    cursor = rows[2]
    assert next_match_index(matches, cursor, cursor._line, forward=False) == 0


def test_next_match_index_cursor_past_last_wraps_to_first_forward():
    rows = _make_rows(["aaa_match", "bbb", "ccc_match", "ddd"])
    matches = list(iter_function_matches(rows, "match"))
    # Cursor on "ddd" (line 3): forward -> no match after, wraps to
    # first ("aaa_match" -> match-list index 0).
    cursor = rows[3]
    assert next_match_index(matches, cursor, cursor._line, forward=True) == 0


def test_next_match_index_cursor_before_first_wraps_to_last_backward():
    rows = _make_rows(["aaa", "bbb_match", "ccc_match", "ddd"])
    matches = list(iter_function_matches(rows, "match"))
    # Cursor on "aaa" (line 0): backward -> no match before, wraps to
    # last (match-list index 1 = "ccc_match").
    cursor = rows[0]
    assert next_match_index(matches, cursor, cursor._line, forward=False) == 1


def test_next_match_index_no_cursor_returns_head_or_tail():
    rows = _make_rows(["aaa_match", "bbb_match"])
    matches = list(iter_function_matches(rows, "match"))
    assert next_match_index(matches, None, -1, forward=True) == 0
    assert next_match_index(matches, None, -1, forward=False) == 1
