"""Regression tests for `kardashev_cli.mcp_server.apply_smart_patch` correctness."""

from __future__ import annotations

import textwrap
from pathlib import Path

from kardashev_cli.mcp_server import apply_smart_patch


def _write_duplicate_block_file(tmp_path: Path) -> Path:
    """A file with the same two-line block appearing at lines 6-7 and again at lines 11-12."""
    content = textwrap.dedent(
        """\
        def alpha():
            pass


        def beta():
            x = 1
            y = 2
            return x + y


        def gamma():
            x = 1
            y = 2
            return x - y
        """
    )
    fpath = tmp_path / "dup.py"
    fpath.write_text(content, encoding="utf-8")
    return fpath


def test_apply_smart_patch_uses_line_hint_to_disambiguate_duplicate_blocks(tmp_path: Path) -> None:
    fpath = _write_duplicate_block_file(tmp_path)

    res = apply_smart_patch(
        target_file=str(fpath),
        target_content="    x = 1\n    y = 2",
        replacement_content="    x = 10\n    y = 20",
        start_line=10,
        end_line=12,
    )

    assert res["applied"] is True
    # The second (gamma) occurrence, not the first (beta) one, must have been edited.
    assert res["start_line"] == 12
    assert res["end_line"] == 13

    updated = fpath.read_text(encoding="utf-8")
    lines = updated.splitlines()
    # beta() body is untouched.
    assert lines[5] == "    x = 1"
    assert lines[6] == "    y = 2"
    # gamma() body was edited.
    assert lines[11] == "    x = 10"
    assert lines[12] == "    y = 20"


def test_apply_smart_patch_reports_actual_edited_span_not_caller_hint(tmp_path: Path) -> None:
    fpath = _write_duplicate_block_file(tmp_path)

    # Caller's hint is intentionally imprecise (off by one line) but still close enough to
    # disambiguate towards the first occurrence; the tool must report where it ACTUALLY wrote,
    # not blindly echo back the caller-supplied start_line/end_line.
    res = apply_smart_patch(
        target_file=str(fpath),
        target_content="    x = 1\n    y = 2",
        replacement_content="    x = 99\n    y = 98",
        start_line=5,
        end_line=8,
    )

    assert res["applied"] is True
    assert (res["start_line"], res["end_line"]) == (6, 7)
    updated = fpath.read_text(encoding="utf-8")
    assert "x = 99" in updated
    assert "x = 1\n    y = 2\n    return x - y" in updated  # gamma() untouched


def test_apply_smart_patch_refuses_ambiguous_default_hint(tmp_path: Path) -> None:
    fpath = _write_duplicate_block_file(tmp_path)
    original = fpath.read_text(encoding="utf-8")

    # No start_line/end_line hint given (defaults), and the block is duplicated: must refuse
    # rather than silently editing the first occurrence.
    res = apply_smart_patch(
        target_file=str(fpath),
        target_content="    x = 1\n    y = 2",
        replacement_content="    x = 0\n    y = 0",
    )

    assert res["applied"] is False
    assert "candidate_line_spans" in res
    assert len(res["candidate_line_spans"]) == 2
    # File must be untouched.
    assert fpath.read_text(encoding="utf-8") == original


def test_apply_smart_patch_refuses_equidistant_hint(tmp_path: Path) -> None:
    fpath = _write_duplicate_block_file(tmp_path)
    original = fpath.read_text(encoding="utf-8")

    # beta's block ends at line 7, gamma's starts at line 12. A [9, 10] hint window sits
    # exactly between them: distance to beta is 9-7=2, distance to gamma is 12-10=2 - a tie.
    res = apply_smart_patch(
        target_file=str(fpath),
        target_content="    x = 1\n    y = 2",
        replacement_content="    x = 0\n    y = 0",
        start_line=9,
        end_line=10,
    )

    assert res["applied"] is False
    assert len(res["candidate_line_spans"]) == 2
    assert fpath.read_text(encoding="utf-8") == original


def test_apply_smart_patch_does_not_write_when_syntax_check_fails(tmp_path: Path) -> None:
    fpath = tmp_path / "simple.py"
    fpath.write_text("def foo():\n    return 1\n", encoding="utf-8")
    original = fpath.read_text(encoding="utf-8")

    res = apply_smart_patch(
        target_file=str(fpath),
        target_content="    return 1",
        replacement_content="    return (",  # invalid syntax
        start_line=1,
        end_line=2,
    )

    assert res["applied"] is False
    assert "syntax_error" in res
    # The file on disk must be completely untouched (checked in-memory before any write).
    assert fpath.read_text(encoding="utf-8") == original


def test_apply_smart_patch_atomic_write_on_success(tmp_path: Path) -> None:
    fpath = tmp_path / "simple2.py"
    fpath.write_text("def foo():\n    return 1\n", encoding="utf-8")

    res = apply_smart_patch(
        target_file=str(fpath),
        target_content="    return 1",
        replacement_content="    return 2",
        start_line=1,
        end_line=2,
    )

    assert res["applied"] is True
    assert fpath.read_text(encoding="utf-8") == "def foo():\n    return 2\n"
    # No leftover temp files in the directory.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".simple2.py.")]
    assert leftovers == []
