"""SSA constant folding and propagation.

The entry point :func:`fold_constants` takes an SSA :class:`Module`
produced by :func:`~compiler_ir.ssa.to_ssa` and returns a brand new SSA
:class:`Module` in which known literal values are propagated along the SSA
def-use chains; the input module is never mutated and shares no mutable
function, block, instruction or phi container with the result.

A value is *known* when it is statically provably a fixed literal:

* a ``Const`` definition is known directly;
* an arithmetic or compare ``BinOp`` folds only when both operands are
  known, and its result becomes known for subsequent definitions;
* a ``Phi`` folds exactly when the literals arriving on *every* reachable
  predecessor edge are all known and identical (a phi's own result is not
  an incoming definition and cannot occur there).

Arithmetic covers ``add``, ``sub``, ``mul``, ``div`` and ``mod``, and
comparisons ``eq``, ``ne``, ``lt``, ``le``, ``gt`` and ``ge``; the int and
bool result types are preserved.  Integer division truncates toward zero
and the remainder satisfies ``a == div(a,b) * b + mod(a,b)``.  A known
zero divisor must not be folded away: the original ``BinOp`` is kept so the
runtime trap and its position in the instruction stream are neither
advanced, swallowed nor rewritten.  A ``Call`` result is always unknown,
and the call instruction together with its relative order is untouched.

Because a loop header phi can look constant on its (constant) entry edge
before the back edge is analyzed, literals are tracked in a monotone
three-level lattice -- unknown, constant ``k``, and non-constant -- solved
to a fixpoint: a phi starts unknown, sinks to its first seen literal and
reaches non-constant as soon as two edges disagree (or an edge carries a
non-constant value).  This is the lattice half of sparse conditional
constant propagation; the reachability half is deliberately absent, so no
blocks are deleted, no constant ``Branch`` becomes a ``Jump``, and
definitions made pointless by folding are left in place for
:func:`~compiler_ir.dce.eliminate_dead_code` to remove.

Folded definitions become a ``Const`` in their *own* SSA slot (the same
value number), so no operand needs rewriting: every existing use already
names that slot, which now holds the literal.  A folded phi loses its phi
node and contributes its ``Const`` ahead of the block's ordinary
instructions (in phi order), matching where the merge value is defined.

Function order, signatures, parameters, block labels and order,
terminators, phi order, the relative order of all instructions and the
existing SSA numbers are all preserved; the output therefore has the same
definition slots as the input.  Applying the pass again gives a
structurally and textually identical fixed point.

A non-:class:`Module` object raises :class:`TypeError`; a non-SSA
:class:`Module` raises :class:`ValueError`.
"""
from .ir_nodes import (
    BinOp,
    Block,
    Branch,
    Call,
    Const,
    Copy,
    Function,
    Jump,
    Module,
    Parameter,
    Phi,
    Return,
    Temp,
)


# Per-value lattice levels (keyed by SSA value id):
#   absent/None      -> unknown (not yet constrained; a phi's initial state)
#   int/bool literal -> known to be exactly that literal
#   _BOTTOM          -> not a compile-time constant
_BOTTOM = object()


def _same_literal(a, b) -> bool:
    # Type-strict equality: the SSA type system never lets an int and a
    # bool meet, but Python's ``True == 1`` must not merge them if a
    # malformed module ever does.
    return type(a) is type(b) and a == b


def fold_constants(module: Module) -> Module:
    """Return a new SSA :class:`Module` with literals folded and propagated.

    The input module is left untouched; the result shares no mutable
    function, block, instruction or phi object with it.  The empty module
    and modules without foldable definitions come back as independent,
    content-equivalent copies.

    :raises TypeError: if ``module`` is not a :class:`Module` instance.
    :raises ValueError: if ``module`` is not in SSA form.
    """
    if not isinstance(module, Module):
        raise TypeError(
            "fold_constants expects a Module, got "
            f"{type(module).__name__}"
        )
    if not getattr(module, "ssa", False):
        raise ValueError("fold_constants expects an SSA Module")
    return Module(
        [_Folder(func).build() for func in module.functions], ssa=True
    )


# --------------------------------------------------------------------------
# Arithmetic / comparison semantics (truncated toward zero, like the runtime)
# --------------------------------------------------------------------------


def _trunc_div(a: int, b: int) -> int:
    """Integer division truncated toward zero.

    Only called for a provably non-zero divisor; a zero divisor is handled
    by the caller so the faulting instruction stays in the stream.
    """
    quotient = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        quotient = -quotient
    return quotient


def _trunc_mod(a: int, b: int) -> int:
    # a == div(a,b) * b + mod(a,b)
    return a - _trunc_div(a, b) * b


_ARITH = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": _trunc_div,
    "mod": _trunc_mod,
}

_COMPARE = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


# --------------------------------------------------------------------------
# Per-function fixpoint and output construction
# --------------------------------------------------------------------------


class _Folder:
    def __init__(self, func: Function):
        self.func = func
        self.lattice: dict = {}
        # Definition id -> user nodes that re-read it, tagged by kind.
        self.users: dict = {}
        self.worklist: list = []

    # -- lattice ------------------------------------------------------------

    def get(self, value):
        return self.lattice.get(id(value))

    def constrain(self, value, level) -> bool:
        """Monotonically move ``value`` toward ``level``.

        Returns whether its level changed.  ``None`` (unknown) never
        constrains anything; two disagreeing literals collapse to
        ``_BOTTOM``.
        """
        if level is None:
            return False
        current = self.lattice.get(id(value))
        if current is _BOTTOM:
            return False
        if current is None:
            self.lattice[id(value)] = level
            return True
        if level is not _BOTTOM and _same_literal(current, level):
            return False
        self.lattice[id(value)] = _BOTTOM
        return True

    def note_user(self, operand, user) -> None:
        self.users.setdefault(id(operand), []).append(user)

    def enqueue_users(self, value) -> None:
        self.worklist.extend(self.users.get(id(value), ()))

    # -- transfer functions -------------------------------------------------

    def phi_level(self, phi: Phi):
        """Join over every reachable incoming edge."""
        result = None
        for value in phi.entries.values():
            level = self.get(value)
            if level is _BOTTOM:
                return _BOTTOM
            if level is None:
                continue
            if result is None:
                result = level
            elif not _same_literal(result, level):
                return _BOTTOM
        return result

    def binop_level(self, ins: BinOp):
        left = self.get(ins.left)
        right = self.get(ins.right)
        if left is None or right is None:
            return None
        if left is _BOTTOM or right is _BOTTOM:
            return _BOTTOM
        if ins.kind == "arith":
            if ins.operator in ("div", "mod") and right == 0:
                # The operands are known but the operation traps: keep the
                # BinOp (its result is never observed), so mark the result
                # non-constant rather than folding the fault away.
                return _BOTTOM
            return _ARITH[ins.operator](left, right)
        return _COMPARE[ins.operator](left, right)

    # -- fixpoint ------------------------------------------------------------

    def solve(self) -> None:
        # Parameters are never statically known.
        for param in self.func.params:
            self.constrain(param.temp, _BOTTOM)

        # Register users and seed every definition.  Traversal order is the
        # deterministic numbering order (blocks in id order, phis before
        # instructions); it affects only iteration count, not the fixpoint.
        for block in self.func.blocks:
            for phi in block.phis:
                for value in phi.entries.values():
                    self.note_user(value, ("phi", phi))
                self.worklist.append(("phi", phi))
            for ins in block.instructions:
                if isinstance(ins, Const):
                    self.worklist.append(("const", ins))
                elif isinstance(ins, BinOp):
                    self.note_user(ins.left, ("binop", ins))
                    self.note_user(ins.right, ("binop", ins))
                    self.worklist.append(("binop", ins))
                elif isinstance(ins, Call):
                    # Call results are always unknown.
                    self.constrain(ins.dest, _BOTTOM)
                elif isinstance(ins, Copy):
                    self.note_user(ins.src, ("copy", ins))
                    self.worklist.append(("copy", ins))

        while self.worklist:
            kind, node = self.worklist.pop()
            if kind == "const":
                changed = self.constrain(node.dest, node.value)
            elif kind == "binop":
                changed = self.constrain(node.dest, self.binop_level(node))
            elif kind == "copy":
                changed = self.constrain(node.dest, self.get(node.src))
            else:
                changed = self.constrain(node.dest, self.phi_level(node))
            if changed:
                self.enqueue_users(node.dest)

    def is_constant(self, value):
        """The settled literal for ``value``, or None if not a constant."""
        level = self.lattice.get(id(value))
        if level is None or level is _BOTTOM:
            return None
        return level

    # -- output --------------------------------------------------------------

    def build(self) -> Function:
        self.solve()

        new_blocks = [Block(block.id) for block in self.func.blocks]
        block_map = dict(zip(self.func.blocks, new_blocks))

        # One fresh Temp per original definition, preserving original ids;
        # every existing operand keeps pointing at the same slot.
        value_map: dict = {}

        def fresh(value: Temp) -> Temp:
            mapped = value_map.get(id(value))
            if mapped is None:
                mapped = Temp(value.id, value.type)
                value_map[id(value)] = mapped
            return mapped

        for param in self.func.params:
            fresh(param.temp)
        for block in self.func.blocks:
            for phi in block.phis:
                fresh(phi.dest)
            for ins in block.instructions:
                fresh(ins.dest)

        def map_value(value):
            if value is None:
                return None
            return value_map[id(value)]

        for block, new_block in zip(self.func.blocks, new_blocks):
            # Folded phis become leading Consts (in phi order); surviving
            # phis keep their entries and predecessor order.
            for phi in block.phis:
                literal = self.is_constant(phi.dest)
                if literal is not None:
                    new_block.instructions.append(
                        Const(map_value(phi.dest), literal)
                    )
                else:
                    entries = {
                        block_map[pred]: map_value(value)
                        for pred, value in sorted(
                            phi.entries.items(), key=lambda item: item[0].id
                        )
                    }
                    new_block.phis.append(
                        Phi(map_value(phi.dest), entries)
                    )

            for ins in block.instructions:
                new_block.instructions.append(
                    self._clone_instruction(ins, map_value)
                )

            term = block.terminator
            if isinstance(term, Return):
                new_block.terminator = Return(map_value(term.value))
            elif isinstance(term, Jump):
                new_block.terminator = Jump(block_map[term.target])
            elif isinstance(term, Branch):
                new_block.terminator = Branch(
                    map_value(term.cond),
                    block_map[term.true_target],
                    block_map[term.false_target],
                )

        new_params = [
            Parameter(param.name, param.slot, map_value(param.temp))
            for param in self.func.params
        ]

        return Function(
            self.func.name,
            new_params,
            self.func.ret_type,
            list(self.func.locals),
            new_blocks,
            block_map[self.func.entry],
            ssa=True,
        )

    def _clone_instruction(self, ins, map_value):
        if isinstance(ins, Const):
            return Const(map_value(ins.dest), ins.value)
        if isinstance(ins, BinOp):
            literal = self.is_constant(ins.dest)
            if literal is not None:
                return Const(map_value(ins.dest), literal)
            return BinOp(
                map_value(ins.dest), ins.operator,
                map_value(ins.left), map_value(ins.right),
                ins.kind, ins.type,
            )
        if isinstance(ins, Call):
            return Call(
                map_value(ins.dest), ins.name,
                [map_value(arg) for arg in ins.args], ins.type,
            )
        # Copy cannot survive to_ssa; handled defensively as pure aliasing.
        if isinstance(ins, Copy):
            return Copy(map_value(ins.dest), map_value(ins.src))
        raise AssertionError(f"unknown instruction: {ins!r}")  # pragma: no cover
