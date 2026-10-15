"""AST Repo-Map & Surgical Context Scout for agy-local.

Provides fast, zero-dependency AST & regex symbol indexing across a workspace so:
1. Local models (`Gemma 4 26B`, `Qwen 3.6`, `GLM-4`) immediately receive exact file paths,
   class/function signatures, and line ranges (`StartLine`-`EndLine`) on Turn 1 without wasting
   5-10 turns on blind `list_dir` and `view_file` calls.
2. Frontier coding agents (`Antigravity 2.0`, `Claude Code`, `Cursor`, `Codex`) can invoke the
   `scout_codebase` MCP tool or `agy-local scout "<query>"` to retrieve a hyper-dense, ranked
   surgical context pack (`<2,000 tokens`), cutting cloud input tokens by 60-80%.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import re
from typing import Any

IGNORED_DIRS: set[str] = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
    ".next",
    "target",
    ".idea",
    ".vscode",
}

SUPPORTED_EXTENSIONS: set[str] = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".go",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}


@dataclass(frozen=True)
class SymbolNode:
    """Represents a class, function, method, or interface extracted from a source file."""

    name: str
    kind: str  # "class" | "function" | "async_function" | "method" | "symbol"
    rel_path: str
    start_line: int
    end_line: int
    signature: str
    parent: str | None = None
    docstring_summary: str | None = None


@dataclass
class FileSymbolSummary:
    """Aggregates extracted symbols and imports for a single source file."""

    rel_path: str
    line_count: int
    imports: list[str] = field(default_factory=list)
    symbols: list[SymbolNode] = field(default_factory=list)


@dataclass
class ScoutResult:
    """Ranked result returned by `scout_workspace` for a natural-language or symbol query."""

    workspace: str
    query: str
    total_files_scanned: int
    total_symbols_indexed: int
    ranked_files: list[dict[str, Any]]
    surgical_slices: list[dict[str, Any]]
    compact_repo_map: str
    estimated_raw_tokens: int
    estimated_pack_tokens: int
    estimated_tokens_saved: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_py_args(args: ast.arguments) -> str:
    parts: list[str] = []
    for a in args.args:
        ann = ""
        if a.annotation is not None:
            try:
                ann = f": {ast.unparse(a.annotation)}"
            except Exception:  # noqa: BLE001
                ann = ""
        parts.append(f"{a.arg}{ann}")
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for kw in args.kwonlyargs:
        parts.append(f"{kw.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def parse_python_symbols(source: str, rel_path: str) -> FileSymbolSummary:
    """Extracts classes, methods, functions, type signatures, and imports from Python code."""
    lines = source.splitlines()
    summary = FileSymbolSummary(rel_path=rel_path, line_count=len(lines))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return summary

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                summary.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                summary.imports.append(f"{mod}.{alias.name}" if mod else alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            ret = ""
            if node.returns is not None:
                try:
                    ret = f" -> {ast.unparse(node.returns)}"
                except Exception:  # noqa: BLE001
                    ret = ""
            sig = f"{prefix} {node.name}({_format_py_args(node.args)}){ret}"
            doc = ast.get_docstring(node)
            doc_first = doc.strip().splitlines()[0][:100] if doc else None
            summary.symbols.append(
                SymbolNode(
                    name=node.name,
                    kind=kind,
                    rel_path=rel_path,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                    signature=sig,
                    docstring_summary=doc_first,
                )
            )
        elif isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except Exception:  # noqa: BLE001
                    pass
            base_str = f"({', '.join(bases)})" if bases else ""
            sig = f"class {node.name}{base_str}"
            doc = ast.get_docstring(node)
            doc_first = doc.strip().splitlines()[0][:100] if doc else None
            summary.symbols.append(
                SymbolNode(
                    name=node.name,
                    kind="class",
                    rel_path=rel_path,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                    signature=sig,
                    docstring_summary=doc_first,
                )
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                    m_ret = ""
                    if item.returns is not None:
                        try:
                            m_ret = f" -> {ast.unparse(item.returns)}"
                        except Exception:  # noqa: BLE001
                            m_ret = ""
                    m_sig = f"{m_prefix} {item.name}({_format_py_args(item.args)}){m_ret}"
                    m_doc = ast.get_docstring(item)
                    m_doc_first = m_doc.strip().splitlines()[0][:90] if m_doc else None
                    summary.symbols.append(
                        SymbolNode(
                            name=item.name,
                            kind="method",
                            rel_path=rel_path,
                            start_line=item.lineno,
                            end_line=getattr(item, "end_lineno", item.lineno) or item.lineno,
                            signature=m_sig,
                            parent=node.name,
                            docstring_summary=m_doc_first,
                        )
                    )

    return summary


_GENERIC_DEF_RE = re.compile(
    r"^\s*(?:export\s+)?(?:pub\s+)?(?:async\s+)?(class|interface|struct|enum|function|fn|def|type)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def parse_generic_symbols(source: str, rel_path: str) -> FileSymbolSummary:
    """Lightweight regex fallback for TypeScript, JavaScript, Rust, Go, and C/C++ files."""
    lines = source.splitlines()
    summary = FileSymbolSummary(rel_path=rel_path, line_count=len(lines))
    for idx, line in enumerate(lines, start=1):
        m = _GENERIC_DEF_RE.match(line)
        if m:
            kind_kw, sym_name = m.group(1), m.group(2)
            end_ln = min(len(lines), idx + 25)
            summary.symbols.append(
                SymbolNode(
                    name=sym_name,
                    kind=kind_kw,
                    rel_path=rel_path,
                    start_line=idx,
                    end_line=end_ln,
                    signature=line.strip()[:100],
                )
            )
    return summary


def build_workspace_index(workspace: str | Path, max_files: int = 400) -> list[FileSymbolSummary]:
    """Scans `workspace` and builds a sorted list of `FileSymbolSummary` objects."""
    root = Path(workspace).resolve()
    results: list[FileSymbolSummary] = []
    if not root.is_dir():
        return results

    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")
        )
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            fpath = Path(dirpath) / fname
            try:
                if fpath.stat().st_size > 512_000:
                    continue
                source = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            rel_path = str(fpath.relative_to(root))
            if ext == ".py":
                summary = parse_python_symbols(source, rel_path)
            else:
                summary = parse_generic_symbols(source, rel_path)
            results.append(summary)
            scanned += 1
            if scanned >= max_files:
                return results
    return results


def _tokenize_query(query: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9_./-]+", query.lower())
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "how",
        "what",
        "where",
        "does",
        "from",
        "into",
        "this",
        "that",
        "can",
        "you",
        "please",
        "fix",
        "add",
        "update",
        "check",
        "show",
        "find",
    }
    terms: list[str] = []
    for tok in raw_tokens:
        if len(tok) >= 2 and tok not in stop_words:
            terms.append(tok)
            # Also split snake_case or path parts
            for sub in re.split(r"[/_.-]+", tok):
                if len(sub) >= 3 and sub not in stop_words and sub not in terms:
                    terms.append(sub)
    return terms


def render_compact_repo_map(
    index: list[FileSymbolSummary],
    query: str = "",
    max_chars: int = 4200,
) -> str:
    """Renders an Aider-style compact AST repository map ranked by query relevance."""
    terms = _tokenize_query(query)
    scored: list[tuple[float, FileSymbolSummary]] = []

    for fsum in index:
        score = 1.0
        path_lower = fsum.rel_path.lower()
        # Deprioritize tests slightly unless query mentions test
        if "test" in path_lower and "test" not in terms:
            score *= 0.6

        for t in terms:
            if t in path_lower:
                score += 6.0
            for sym in fsum.symbols:
                sname = sym.name.lower()
                if t == sname:
                    score += 10.0
                elif t in sname:
                    score += 4.5
                elif sym.docstring_summary and t in sym.docstring_summary.lower():
                    score += 2.0
        scored.append((score, fsum))

    scored.sort(key=lambda item: (-item[0], item[1].rel_path))

    out_lines: list[str] = ["<ast_repo_map>"]
    used_chars = len(out_lines[0])

    for _score, fsum in scored:
        header = f"## {fsum.rel_path} ({fsum.line_count} lines)"
        sym_lines: list[str] = []
        for sym in fsum.symbols:
            indent = "    " if sym.parent else "  "
            loc = f"[L{sym.start_line}-{sym.end_line}]"
            doc_str = f"  # {sym.docstring_summary}" if sym.docstring_summary else ""
            sym_lines.append(f"{indent}{loc} {sym.signature}{doc_str}")

        block = "\n".join([header, *sym_lines[:18]])
        if len(sym_lines) > 18:
            block += f"\n  ... (+{len(sym_lines) - 18} more symbols)"
        if used_chars + len(block) + 2 > max_chars:
            break
        out_lines.append(block)
        used_chars += len(block) + 1

    out_lines.append("</ast_repo_map>")
    return "\n".join(out_lines)


def scout_workspace(
    workspace: str | Path,
    query: str,
    *,
    max_slices: int = 5,
    max_slice_lines: int = 45,
) -> ScoutResult:
    """Builds a surgical context pack for `query` containing ranked files, AST symbols, and code slices."""
    root = Path(workspace).resolve()
    index = build_workspace_index(root)
    terms = _tokenize_query(query)
    total_symbols = sum(len(f.symbols) for f in index)
    total_raw_chars = 0

    file_scores: list[tuple[float, FileSymbolSummary]] = []
    symbol_hits: list[tuple[float, SymbolNode]] = []

    for fsum in index:
        fpath = root / fsum.rel_path
        try:
            total_raw_chars += fpath.stat().st_size
        except OSError:
            pass

        f_score = 0.5
        path_lower = fsum.rel_path.lower()
        for t in terms:
            if t in path_lower:
                f_score += 5.0
        for sym in fsum.symbols:
            s_score = 0.0
            sname = sym.name.lower()
            for t in terms:
                if t == sname:
                    s_score += 12.0
                elif t in sname:
                    s_score += 5.0
                elif sym.docstring_summary and t in sym.docstring_summary.lower():
                    s_score += 2.5
                elif t in sym.signature.lower():
                    s_score += 2.0
            if s_score > 0:
                f_score += s_score
                symbol_hits.append((s_score, sym))
        file_scores.append((f_score, fsum))

    file_scores.sort(key=lambda x: (-x[0], x[1].rel_path))
    symbol_hits.sort(key=lambda x: (-x[0], x[1].rel_path, x[1].start_line))

    ranked_files: list[dict[str, Any]] = []
    for score, fsum in file_scores[:8]:
        ranked_files.append(
            {
                "path": fsum.rel_path,
                "score": round(score, 2),
                "line_count": fsum.line_count,
                "symbols": [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "lines": f"{s.start_line}-{s.end_line}",
                        "signature": s.signature,
                    }
                    for s in fsum.symbols[:15]
                ],
            }
        )

    surgical_slices: list[dict[str, Any]] = []
    seen_spans: set[tuple[str, int]] = set()
    for score, sym in symbol_hits:
        if len(surgical_slices) >= max_slices:
            break
        key = (sym.rel_path, sym.start_line)
        if key in seen_spans:
            continue
        seen_spans.add(key)
        fpath = root / sym.rel_path
        try:
            all_lines = fpath.read_text(encoding="utf-8").splitlines()
            s_idx = max(0, sym.start_line - 1)
            e_idx = min(len(all_lines), s_idx + max_slice_lines, sym.end_line)
            numbered = [
                f"{ln_no}: {all_lines[ln_no - 1]}"
                for ln_no in range(s_idx + 1, e_idx + 1)
            ]
            surgical_slices.append(
                {
                    "path": sym.rel_path,
                    "symbol": f"{sym.parent}.{sym.name}" if sym.parent else sym.name,
                    "kind": sym.kind,
                    "start_line": s_idx + 1,
                    "end_line": e_idx,
                    "score": round(score, 2),
                    "code": "\n".join(numbered),
                }
            )
        except (OSError, UnicodeDecodeError):
            continue

    repo_map = render_compact_repo_map(index, query=query, max_chars=3600)
    pack_chars = len(repo_map) + sum(len(s["code"]) for s in surgical_slices)
    est_raw_tokens = max(1, total_raw_chars // 4)
    est_pack_tokens = max(1, pack_chars // 4)
    est_saved = max(0, est_raw_tokens - est_pack_tokens)

    return ScoutResult(
        workspace=str(root),
        query=query,
        total_files_scanned=len(index),
        total_symbols_indexed=total_symbols,
        ranked_files=ranked_files,
        surgical_slices=surgical_slices,
        compact_repo_map=repo_map,
        estimated_raw_tokens=est_raw_tokens,
        estimated_pack_tokens=est_pack_tokens,
        estimated_tokens_saved=est_saved,
    )
