"""Derived channels: a small expression language over other channels, and the evaluation plan."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .settings import SLOT_MINUTES, ChannelSpec


def _mask(x: Any, m: Any) -> Any:
    """x where m > 0, missing (NaN) elsewhere, e.g. mask(vis, cos_sza - 0.1) for daylight only."""
    with np.errstate(invalid="ignore"):
        return np.where(np.asarray(m) > 0, x, np.float32(np.nan))


# f(x, minutes): x at an earlier slot, or reduced over the slots of the last `minutes` (current one included)
TIME_FUNCTIONS = ("lag", "sum_past", "mean_past", "min_past", "max_past")
ELEMENTWISE_FUNCTIONS = {
    "abs": (1, np.abs),
    "sqrt": (1, np.sqrt),
    "log1p": (1, np.log1p),
    "exp": (1, np.exp),
    "minimum": (2, np.minimum),
    "maximum": (2, np.maximum),
    "mask": (2, _mask),
}
FUNCTION_NAMES = set(TIME_FUNCTIONS) | set(ELEMENTWISE_FUNCTIONS) | {"clip"}
BINARY_OPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
PAST_REDUCERS = {"sum_past": np.sum, "mean_past": np.mean, "min_past": np.min, "max_past": np.max}


def _number(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _number(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    return None


def _time_function_slots(node: ast.Call) -> int:
    return int(_number(node.args[1])) // SLOT_MINUTES


def parse_expression(text: str, where: str) -> ast.Expression:
    """Parses a channel expression, allowing only arithmetic on channel names, numbers and FUNCTION_NAMES."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"{where}: cannot parse {text!r}: {e.msg}") from None

    def check(node: ast.AST) -> None:
        if _number(node) is not None or isinstance(node, ast.Name):
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return check(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
            check(node.left)
            check(node.right)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            f, args = node.func.id, node.args
            if f in TIME_FUNCTIONS:
                minutes = _number(args[1]) if len(args) == 2 else None
                if minutes is None or minutes <= 0 or minutes % SLOT_MINUTES:
                    raise ValueError(f"{where}: {f}(x, minutes) needs minutes as a positive multiple of {SLOT_MINUTES}")
                return check(args[0])
            if f == "clip":
                if len(args) != 3 or _number(args[1]) is None or _number(args[2]) is None:
                    raise ValueError(f"{where}: clip(x, low, high) needs numbers for low and high")
                return check(args[0])
            if f in ELEMENTWISE_FUNCTIONS:
                if len(args) != ELEMENTWISE_FUNCTIONS[f][0]:
                    raise ValueError(f"{where}: {f} takes {ELEMENTWISE_FUNCTIONS[f][0]} argument(s)")
                for a in args:
                    check(a)
                return
            raise ValueError(f"{where}: unknown function {f!r}; available: {sorted(FUNCTION_NAMES)}")
        raise ValueError(f"{where}: unsupported syntax {ast.unparse(node)!r} (use + - * / **, numbers, channel names and functions)")

    check(tree.body)
    return tree


def expression_refs(node: ast.AST, offset: int = 0) -> Iterator[Tuple[str, int]]:
    """(channel name, slots back) for every channel the expression reads."""
    if isinstance(node, ast.Expression):
        yield from expression_refs(node.body, offset)
    elif isinstance(node, ast.Name):
        yield node.id, offset
    elif isinstance(node, ast.Call):
        f = node.func.id
        if f in TIME_FUNCTIONS:
            n = _time_function_slots(node)
            yield from expression_refs(node.args[0], offset + (n if f == "lag" else n - 1))
        else:
            for a in node.args:
                yield from expression_refs(a, offset)
    else:
        for child in ast.iter_child_nodes(node):
            yield from expression_refs(child, offset)


@dataclass
class Plan:
    order: List[ChannelSpec]  # every channel after the channels it reads
    exprs: Dict[str, ast.Expression]
    depth: Dict[str, int]  # oldest slot (back from the current one) at which each channel is read
    lookback: int  # slots computed before the day so the first output slot has full history
    lookahead: int  # GLM slots after the last issue time that the labels need
    # Channels computed at every slot: read at earlier slots (lag, *_past), carrying values forward
    # (fill_gaps), or read by such a channel. The others are only computed at output slots.
    every_slot: frozenset


def make_plan(channels: Tuple[ChannelSpec, ...], label_end_slots: List[int]) -> Plan:
    by_name = {c.name: c for c in channels}
    exprs = {c.name: parse_expression(c.expr, f"channel {c.name!r}") for c in channels if c.kind == "expr"}
    refs = {name: list(expression_refs(tree)) for name, tree in exprs.items()}
    for name, rs in refs.items():
        missing = sorted({r for r, _ in rs if r not in by_name})
        if missing:
            raise ValueError(f"channel {name!r} refers to unknown channels {missing}")

    order: List[ChannelSpec] = []
    state: Dict[str, int] = {}  # 1 = visiting, 2 = done

    def visit(name: str, path: List[str]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise ValueError(f"channel expressions form a cycle: {' -> '.join(path + [name])}")
        state[name] = 1
        for r, _ in refs.get(name, []):
            visit(r, path + [name])
        state[name] = 2
        order.append(by_name[name])

    for c in channels:
        visit(c.name, [])

    depth = {c.name: 0 for c in channels}
    for rs in refs.values():
        for r, off in rs:
            depth[r] = max(depth[r], off)

    need: Dict[str, int] = {}
    for c in order:  # dependencies first, so need[r] is known
        if c.kind == "product":
            need[c.name] = c.fill_gaps_slots
        elif c.kind in ("glm", "solar"):
            need[c.name] = 0
        else:
            need[c.name] = max((off + need[r] for r, off in refs[c.name]), default=0)

    every = {c.name for c in channels if c.kind == "product" and c.fill_gaps_slots}
    every |= {r for rs in refs.values() for r, off in rs if off > 0}
    for c in reversed(order):  # readers before the channels they read
        if c.name in every:
            every |= {r for r, _ in refs.get(c.name, [])}
    return Plan(order, exprs, depth, max(need.values()), max(label_end_slots, default=0), frozenset(every))


class Evaluator:
    """Evaluates a parsed expression at slot s from the channel frames cached so far."""

    def __init__(self, cache: Dict[str, Dict[int, np.ndarray]], shape: Tuple[int, int]):
        self.cache = cache
        self.missing = np.full(shape, np.nan, dtype=np.float32)
        self.missing.setflags(write=False)

    def __call__(self, node: ast.AST, s: int) -> Any:
        v = _number(node)
        if v is not None:
            return np.float32(v)
        if isinstance(node, ast.Expression):
            return self(node.body, s)
        if isinstance(node, ast.Name):
            # Slots before the computed range only occur while warming up; those outputs are never written
            return self.cache[node.id].get(s, self.missing)
        if isinstance(node, ast.UnaryOp):
            x = self(node.operand, s)
            return -x if isinstance(node.op, ast.USub) else x
        if isinstance(node, ast.BinOp):
            return BINARY_OPS[type(node.op)](self(node.left, s), self(node.right, s))
        f, args = node.func.id, node.args
        if f == "lag":
            return self(args[0], s - _time_function_slots(node))
        if f in PAST_REDUCERS:
            frames = [self(args[0], s - i) for i in range(_time_function_slots(node))]
            return PAST_REDUCERS[f](np.stack(np.broadcast_arrays(*frames)), axis=0)  # NaN in any slot -> NaN
        if f == "clip":
            return np.clip(self(args[0], s), _number(args[1]), _number(args[2]))
        return ELEMENTWISE_FUNCTIONS[f][1](*(self(a, s) for a in args))
