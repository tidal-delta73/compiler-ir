"""Loop-invariant code motion on pruned SSA form.

The entry point :func:`hoist_loop_invariants` takes an SSA
:class:`Module` produced by :func:`~compiler_ir.ssa.to_ssa` and returns a
brand new SSA :class:`Module` in which safe, loop-invariant instructions
are moved out of their loops; the input module is never mutated and shares
no mutable function, block, instruction or phi container with the result.

Loops and the shape gate
------------------------

A *back edge* is an edge ``tail -> header`` whose header dominates its
tail; the reachable *natural loop* is the header plus every block that can
reach the tail without passing through the header (several back edges into
one header share one loop -- their bodies are unioned).  Code is moved out
of a natural loop only when it has exactly the shape needed for the move
to be unconditionally safe:

* the header has exactly one loop-external predecessor ``p``;
* that predecessor branches only to the header (its sole successor is
  the header).

Loops not meeting this shape are left exactly as they are, and no new
basic block is ever created.

Hoistable instructions
----------------------

An instruction is hoisted only when it is safe to execute unconditionally
on the single in-edge and every iteration computes the same value:

* ``Const`` is always invariant;
* an ``add``/``sub``/``mul`` arithmetic ``BinOp`` and every comparison
  ``BinOp`` are hoisted once all of their operands are defined outside the
  loop or were themselves hoisted in the same processing of this loop;
* ``Phi``, ``Call`` and ``Copy`` are never moved, and neither are ``div``
  or ``mod``: a zero-iteration loop must neither gain a call nor execute a
  (potentially faulting) division or remainder early.

Instructions are placed in the unique external predecessor block,
immediately before its terminator.  Nested loops are processed inner loop
before outer loop, so an instruction invariant to an inner loop migrates
to that loop's preheader first; each instruction moves at most once.
Moved instructions keep the relative order given by the original block
order and the original within-block instruction order, so the transform is
deterministic and the result a fixed point: applying the pass again changes
nothing structurally or textually.

Function order, signatures, parameters, block labels and order,
terminators, phi nodes and their predecessor order, and the existing SSA
numbers are all preserved -- moved instructions keep their own value slot,
so no operand is rewritten.

A non-:class:`Module` object raises :class:`TypeError`; a non-SSA
:class:`Module` raises :class:`ValueError`.  The empty module and modules
without a hoistable instruction come back as independent,
content-equivalent copies.
"""
from .cfg import analyze_function
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

#: Arithmetic operators that cannot trap and may be speculatively executed.
_SAFE_ARITH = frozenset(("add", "sub", "mul"))


def hoist_loop_invariants(module: Module) -> Module:
    """Return a new SSA :class:`Module` with loop invariants hoisted.

    The input module is left untouched; the result shares no mutable
    function, block, instruction or phi object with it.  Modules without a
    hoistable loop come back as independent, content-equivalent copies.

    :raises TypeError: if ``module`` is not a :class:`Module` instance.
    :raises ValueError: if ``module`` is not in SSA form.
    """
    if not isinstance(module, Module):
        raise TypeError(
            "hoist_loop_invariants expects a Module, got "
            f"{type(module).__name__}"
        )
    if not getattr(module, "ssa", False):
        raise ValueError("hoist_loop_invariants expects an SSA Module")
    return Module(
        [_Hoister(func).run() for func in module.functions], ssa=True
    )


# --------------------------------------------------------------------------
# Natural-loop detection
# --------------------------------------------------------------------------


def _natural_loops(flow):
    """Return the reachable natural loops of one function.

    A back edge is an edge ``tail -> header`` with the header dominating
    the tail; the natural loop is the header plus the blocks that reach a
    back-edge tail without going through the header.  Back edges sharing a
    header give one loop (their bodies union).  The result is ordered
    deterministically by header position, body size and body positions.
    """
    idom = flow.idom
    succs = flow.succs
    preds = flow.preds
    index = flow.index

    def dominates(header: Block, block: Block) -> bool:
        node = block
        while True:
            if node is header:
                return True
            if node is idom[node]:
                return False
            node = idom[node]

    # header -> back-edge tails, headers met in reverse postorder
    back_edges: dict = {}
    for tail in flow.rpo:
        for header in succs[tail]:
            if header in idom and dominates(header, tail):
                back_edges.setdefault(header, []).append(tail)

    loops = []
    for header in flow.rpo:
        tails = back_edges.get(header)
        if not tails:
            continue
        body = {header}
        stack = list(tails)
        while stack:
            block = stack.pop()
            if block in body:
                continue
            body.add(block)
            # Predecessors join the loop; the header is already in the body
            # and stops the walk.
            stack.extend(preds[block])
        loops.append((header, body))

    def loop_key(loop):
        _header, body = loop
        return (
            index[_header],
            len(body),
            tuple(sorted(index[b] for b in body)),
        )

    return sorted(loops, key=loop_key)


# --------------------------------------------------------------------------
# Per-function analysis, hoist planning and output construction
# --------------------------------------------------------------------------


class _Hoister:
    def __init__(self, func: Function):
        self.func = func
        self.flow = analyze_function(func)

    # -- loop eligibility ---------------------------------------------------

    def _loop_preheader(self, header: Block, body: frozenset):
        """The unique external predecessor that flows only to the header.

        Returns that block, or ``None`` when the loop fails the shape gate:
        the header must have exactly one predecessor outside the loop body,
        and that predecessor's sole successor must be the header.
        """
        external = [
            pred for pred in self.flow.preds[header] if pred not in body
        ]
        if len(external) != 1:
            return None
        pre = external[0]
        if self.flow.succs[pre] != [header]:
            return None
        return pre

    # -- planning -----------------------------------------------------------

    def _plan(self):
        """Decide every move before constructing the output.

        Returns ``moves`` mapping ``id(instruction.dest)`` to
        ``(origin block, final preheader)``.

        Nested loops are processed inner first (a nested natural-loop body
        is strictly smaller than its enclosing loop's).  An instruction
        hoisted out of an inner loop to a preheader inside an outer loop is
        claimed again while that outer loop is processed, so one pass
        carries an invariant of several loops straight to the outermost
        eligible preheader; the instruction still physically moves once,
        from its origin to that final block.  Every pass thus ends at a
        fixed point: re-running it changes nothing.
        """
        loops = _natural_loops(self.flow)

        # Definition -> current owning block for every in-block
        # definition.  Phi results are registered at their header/merge
        # block: a phi never moves, and an operand defined by a phi inside
        # the loop is loop-variant (not "outside").  Only parameters have
        # no owner; they dominate the whole function.
        owner: dict = {}
        for block in self.func.blocks:
            for phi in block.phis:
                owner[id(phi.dest)] = block
            for ins in block.instructions:
                owner[id(ins.dest)] = block

        # id(instruction.dest) -> (origin block, currently chosen preheader)
        moves: dict = {}

        def defined_outside(operand, body: frozenset) -> bool:
            defining_block = owner.get(id(operand))
            if defining_block is None:
                # Only parameters lack an owner; they are defined on entry
                # and dominate every block.
                return True
            return defining_block not in body

        # Inner first: nested natural-loop bodies are strict subsets, so
        # body size orders containment; ties break on header position.
        for header, body_set in sorted(
            loops, key=lambda loop: (len(loop[1]), self.flow.index[loop[0]])
        ):
            body = frozenset(body_set)
            pre = self._loop_preheader(header, body)
            if pre is None:
                continue

            # Stabilize candidacy to a fixpoint so chains of invariants
            # hoist together.  Block/instruction order is deterministic; in
            # valid SSA a def already dominates its use, but the loop makes
            # the result independent of sweep schedule regardless.
            progress = True
            while progress:
                progress = False
                for block in self.flow.sort(body):
                    for ins in block.instructions:
                        dest_id = id(ins.dest)
                        current = moves.get(dest_id)
                        if current is not None and current[1] is pre:
                            # Already placed in this very preheader during
                            # this loop's fixpoint; an outer loop may still
                            # claim it later.
                            continue
                        if not self._can_hoist(
                            ins, lambda v: defined_outside(v, body)
                        ):
                            continue
                        moves[dest_id] = (
                            block if current is None else current[0], pre)
                        owner[dest_id] = pre
                        progress = True

        return moves

    @staticmethod
    def _can_hoist(ins, defined_outside) -> bool:
        if isinstance(ins, Const):
            return True
        if isinstance(ins, BinOp):
            if ins.kind == "compare":
                allowed = True
            elif ins.kind == "arith":
                allowed = ins.operator in _SAFE_ARITH
            else:
                allowed = False
            return allowed and defined_outside(
                ins.left) and defined_outside(ins.right)
        # Phi is not an instruction (handled via block.phis); Call and Copy
        # are never hoisted.
        return False

    # -- output -------------------------------------------------------------

    def run(self) -> Function:
        moves = self._plan()

        # Moved instructions per preheader, in globally deterministic order:
        # original block position, then original instruction position.
        block_index = {
            block: i for i, block in enumerate(self.func.blocks)
        }
        inserted: dict = {}
        ordered = []
        for block in self.func.blocks:
            for position, ins in enumerate(block.instructions):
                target = moves.get(id(ins.dest))
                if target is not None:
                    _origin, pre = target
                    ordered.append(
                        (block_index[block], position, pre, ins))
        ordered.sort(key=lambda item: (item[0], item[1]))
        moved_dest_ids: set = set()
        for _b, _i, pre, ins in ordered:
            inserted.setdefault(id(pre), []).append(ins)
            moved_dest_ids.add(id(ins.dest))

        new_blocks = [Block(block.id) for block in self.func.blocks]
        block_map = dict(zip(self.func.blocks, new_blocks))

        # One fresh Temp per original definition; original ids are kept and
        # moved instructions keep their own slots, so no operand is
        # rewritten.
        value_map: dict = {}

        def note(value) -> None:
            value_map[id(value)] = Temp(value.id, value.type)

        for param in self.func.params:
            note(param.temp)
        for block in self.func.blocks:
            for phi in block.phis:
                note(phi.dest)
            for ins in block.instructions:
                note(ins.dest)

        def map_value(value):
            if value is None:
                return None
            return value_map[id(value)]

        for block, new_block in zip(self.func.blocks, new_blocks):
            for phi in block.phis:
                entries = {
                    block_map[pred]: map_value(value)
                    for pred, value in sorted(
                        phi.entries.items(), key=lambda item: item[0].id
                    )
                }
                new_block.phis.append(Phi(map_value(phi.dest), entries))

            hoisted_here = inserted.get(id(block), ())

            # Instructions moved OUT of this block are dropped here and
            # re-emitted at their preheader; instructions inserted into
            # this block follow the surviving instructions, immediately
            # before its terminator.
            for ins in block.instructions:
                if id(ins.dest) in moved_dest_ids:
                    continue
                new_block.instructions.append(
                    self._clone_instruction(ins, map_value))
            for ins in hoisted_here:
                new_block.instructions.append(
                    self._clone_instruction(ins, map_value))

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
        if isinstance(ins, Copy):
            return Copy(map_value(ins.dest), map_value(ins.src))
        if isinstance(ins, BinOp):
            return BinOp(
                map_value(ins.dest), ins.operator,
                map_value(ins.left), map_value(ins.right),
                ins.kind, ins.type,
            )
        return Call(
            map_value(ins.dest), ins.name,
            [map_value(arg) for arg in ins.args], ins.type,
        )
