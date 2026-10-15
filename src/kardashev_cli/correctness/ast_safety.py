"""AST allowlisting for executable invariant expressions.

Invariants are short boolean expressions such as ``"out >= 0"`` or
``"not (p > 0) or math.isclose(f(p), 1 - f(-p))"``. They are untrusted text
submitted alongside a concept, so before any invariant expression is ever
sent to the sandboxed executor for evaluation, its AST is checked against a
narrow allowlist: no imports, no dunders, no attribute access except on the
``math`` module, and calls only to ``f``, ``math.*``, or a small set of safe
builtins.
"""

from __future__ import annotations

import ast

SAFE_CALL_NAMES = {"f", "abs", "min", "max", "len", "sum", "all", "any", "isclose", "range", "round"}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UnaryOp,
    ast.Not,
    ast.UAdd,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Index,  # py<3.9 compat node, harmless if absent on this version
    ast.Slice,
    ast.IfExp,
    ast.ListComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Attribute,
)


class InvariantSyntaxError(ValueError):
    """Raised when an invariant expression fails the AST allowlist."""


def extract_param_names(source_code: str, func_name: str) -> list[str]:
    """Return the positional parameter names of ``func_name`` in ``source_code``.

    Parsing only (no execution), so this is safe to run on untrusted source.
    """
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            args = node.args
            names = [a.arg for a in args.posonlyargs] if hasattr(args, "posonlyargs") else []
            names += [a.arg for a in args.args]
            names += [a.arg for a in args.kwonlyargs]
            return names
    raise InvariantSyntaxError(f"function '{func_name}' not found in source")


def validate_invariant_ast(expr: str, allowed_names: set[str]) -> ast.Expression:
    """Parse ``expr`` as a boolean expression and enforce the allowlist.

    ``allowed_names`` must include the concept's parameter names, ``f``, and
    ``out``. Returns the parsed AST on success; raises
    :class:`InvariantSyntaxError` otherwise. Never evaluates the expression.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise InvariantSyntaxError(f"not a valid expression: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise InvariantSyntaxError(f"disallowed syntax: {type(node).__name__}")

        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise InvariantSyntaxError("dunder identifiers are forbidden")
            if node.id not in allowed_names and node.id not in SAFE_CALL_NAMES and node.id != "math":
                raise InvariantSyntaxError(f"name '{node.id}' is not allowed in an invariant")

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise InvariantSyntaxError("dunder/private attributes are forbidden")
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                raise InvariantSyntaxError("attribute access is only allowed on the 'math' module")

        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name):
                if callee.id not in SAFE_CALL_NAMES:
                    raise InvariantSyntaxError(f"call to '{callee.id}' is not allowed")
            elif isinstance(callee, ast.Attribute):
                if not (isinstance(callee.value, ast.Name) and callee.value.id == "math"):
                    raise InvariantSyntaxError("only math.* calls are allowed besides f(...)")
            else:
                raise InvariantSyntaxError("indirect calls are not allowed")

    return tree


def is_free_text(expr: str, allowed_names: set[str]) -> bool:
    """True when ``expr`` cannot be parsed/validated as a safe boolean expression."""
    try:
        validate_invariant_ast(expr, allowed_names)
    except InvariantSyntaxError:
        return True
    return False
