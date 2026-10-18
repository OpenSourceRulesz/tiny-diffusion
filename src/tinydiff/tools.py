"""Toy tools with deterministic behavior, so tool-use correctness is checkable."""

import ast
import operator

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calc(expr):
    """Safely evaluate a basic arithmetic expression."""
    try:
        value = _eval(ast.parse(expr.strip(), mode="eval"))
    except Exception as e:
        return f"error: {e}"
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


KB = {
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
    "capital of brazil": "Brasilia",
    "capital of egypt": "Cairo",
    "capital of canada": "Ottawa",
    "capital of australia": "Canberra",
    "largest planet": "Jupiter",
    "smallest planet": "Mercury",
    "currency of japan": "yen",
    "currency of the uk": "pound",
    "boiling point of water": "100 degrees celsius",
    "speed of light": "299792458 meters per second",
}


def lookup(query):
    """Look up a fact in a tiny knowledge base."""
    return KB.get(query.strip().lower(), "unknown")


TOOLS = {"calc": calc, "lookup": lookup}


def run_call(call_text):
    """Parse 'toolname args...' and execute. Returns the result string."""
    parts = call_text.strip().split(None, 1)
    if not parts:
        return "error: empty call"
    name, args = parts[0], (parts[1] if len(parts) > 1 else "")
    fn = TOOLS.get(name)
    if fn is None:
        return f"error: unknown tool '{name}'"
    try:
        return fn(args)
    except Exception as e:
        return f"error: {e}"
