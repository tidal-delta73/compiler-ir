"""Loop-invariant code motion on pruned SSA form.

The entry point :func:`hoist_loop_invariants` takes an SSA
:class:`Module` produced by :func:`~compiler_ir.ssa.to_ssa` and returns a
brand new SSA :class:`Module` in which safe loop-invariant computations
are moved out of the loops that recompute them; the input module is never
mutated and shares no mutable function, block, instruction or phi
container with the result.

Natural loops
-------------

For every back edge ``tail -> head`` (an edge whose destination
dominates its source, a self edge included) the reachable natural loop
is ``{head}`` plus every block that can reach the tail without passing
through the head.  Several latches branching back to one header are
one loop: their bodies are unioned under that header.

A loop is hoistable only when its header has exactly one predecessor
outside the loop (the *preheader*) and that preheader flows nowhere but
to the header.  Loops of any other shape are left completely untouched,
and no new basic blocks are ever created.

Hoistable definitions
---------------------

Inside a hoistable loop the following definitions move to the
preheader, immediately before its terminator, when every operand is
defined outside the loop, or is itself a definition promoted ahead of
the loop in the same processing:

* a ``Const`` (always invariant);
* an arithmetic ``BinOp`` with operator ``add``, ``sub`` or ``mul``;
* a comparison ``BinOp`` (any comparison operator).

``Phi`` nodes, ``Call`` and ``Copy`` instructions never move, and
neither do ``div``/``mod`` BinOps: this keeps a zero-trip loop from
gaining a call it never made and keeps a division/modulo trap at its
original site and position in the instruction stream.  As a second
guard on the trap *site*, a trap-free ``BinOp`` only leaves a block
when it is positioned after the block's last ``div``/``mod``: removing
an earlier BinOp would otherwise renumber the faulting instruction's
ordinal within that block.  Every moved instruction is pure and
trap-free, so executing it unconditionally on the loop's entry edge
changes no observable behavior.

Nested loops are analyzed inner-most first.  A definition invariant to
an enclosing loop is promoted from the inner loop's candidate set to
the enclosing loop's set before anything moves, so each instruction
physically moves exactly once and lands ahead of the outermost
hoistable loop it can leave.  Promotion never crosses a loop that fails
the preheader shape gate: such a loop keeps every definition it
contains for the whole pass (definitions may still move to preheaders
of hoistable loops nested inside it).  The final placement follows the
original block order and within-block instruction order (which SSA
def-before-use dominance makes a valid schedule), so the result is
unique for a given input.

Function order, signatures, parameters, block labels and order,
terminators, phi nodes and the existing SSA numbers are all preserved
(the pass does not renumber); a module with no hoistable loop comes
back as an independent, content-equivalent copy.  Applying the pass
again changes nothing structurally or textually.

A non-:class:`Module` object raises :class:`TypeError`; a non-SSA
:class:`Module` raises :class:`ValueError`.
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
)

#: Arithmetic operators safe to speculate (never trap).
_HOISTABLE_ARITH = frozenset(("add", "sub", "mul"))


def hoist_loop_invariants(module: Module) -> Module:
    """Return a new SSA :class:`Module` with loop invariants hoisted.

    The input module is left untouched; the result shares no mutable
    function, block, instruction or phi object with it.  The empty
    module and modules without hoistable instructions come back as
    independent, content-equivalent copies.

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
        [_hoist_in_function(func) for func in module.functions], ssa=True
    )


# --------------------------------------------------------------------------
# Natural-loop detection
# --------------------------------------------------------------------------


def _natural_loops(flow):
    """Return the function's distinct reachable natural loops.

    Each item is ``(header, body_set)`` over the analysis's original
    blocks.  Results are deduplicated by block set and ordered
    deterministically: smallest block index in the body first, then set
    size, then the full ascending index tuple.
    """
    index = flow.index
    preds = flow.preds
    idom = flow.idom
    reachable = flow.reachable

    # A back edge tail -> head has the head dominating the tail (a block
    # dominates itself, so loop self-edges count).
    back_edges = []
    for block in flow.reachable_blocks:
        for pred in preds[block]:
            if pred is block or _dominates(idom, block, pred):
                back_edges.append((pred, block))

    # Group by header and union the per-latch bodies: every latch of a
    # header belongs to the same natural loop.
    bodies_by_header: dict = {}
    order_headers: list = []
    for tail, header in back_edges:
        body = {header}
        if tail is not header:
            body.add(tail)
            stack = [tail]
            while stack:
                block = stack.pop()
                for pred in preds[block]:
                    if pred in reachable and pred not in body:
                        body.add(pred)
                        stack.append(pred)
        if header not in bodies_by_header:
            bodies_by_header[header] = set(body)
            order_headers.append(header)
        else:
            bodies_by_header[header].update(body)

    loops = [
        (header, frozenset(bodies_by_header[header]))
        for header in order_headers
    ]

    def loop_key(item):
        _header, body_set = item
        indices = sorted(index[b] for b in body_set)
        return (indices[0], len(indices), tuple(indices))

    return sorted(loops, key=loop_key)


def _dominates(idom: dict, dominator: Block, block: Block) -> bool:
    """Whether ``dominator`` dominates ``block`` given immediate dominators."""
    current = block
    while True:
        if current is dominator:
            return True
        parent = idom[current]
        if parent is current:
            return False
        current = parent


def _preheader(flow, header: Block, body: frozenset):
    """Return the unique-shape loop preheader, or ``None``.

    The header must have exactly one predecessor outside the loop, and
    that block must have the header as its sole (deduplicated) successor.
    """
    outside = [pred for pred in flow.preds[header] if pred not in body]
    if len(outside) != 1:
        return None
    pre = outside[0]
    if flow.succs[pre] != [header]:
        return None
    return pre


# --------------------------------------------------------------------------
# Per-function transform
# --------------------------------------------------------------------------


def _hoist_in_function(func: Function) -> Function:
    flow = analyze_function(func)
    loops = _natural_loops(flow)

    # Clone first; every mutation below applies to the cloned graph, so
    # the input function is never touched.
    new_blocks = [Block(block.id) for block in func.blocks]
    block_map = dict(zip(func.blocks, new_blocks))
    _clone_into(func, new_blocks, block_map)
    block_index = {block_map[b]: i for b, i in flow.index.items()}

    # Loops translated onto the cloned graph: (body, preheader), with a
    # None preheader marking a loop that fails the shape gate.
    tloops = []
    for header, body in loops:
        pre = _preheader(flow, header, body)
        tloops.append((
            frozenset(block_map[b] for b in body),
            None if pre is None else block_map[pre],
        ))

    # Inner-most loops first: a loop whose body is strictly contained in
    # another loop's body is processed before it.  Detection order is the
    # stable tie-breaker.
    def depth(i):
        return sum(1 for j in range(len(loops)) if loops[i][1] < loops[j][1])

    process_order = sorted(
        range(len(loops)), key=lambda i: (-depth(i), i)
    )

    # Original (block position, instruction index) of every cloned
    # instruction, fixed before any movement.
    original_pos = {}
    for block in new_blocks:
        for i, ins in enumerate(block.instructions):
            original_pos[id(ins)] = (block_index[block], i)

    param_defs = {id(param.temp) for param in func.params}

    # One candidate list per hoistable loop; an instruction sits in at
    # most one list at any time.  ``claimed`` notes instructions that
    # already entered some list directly from a loop body.
    candidates = {i: [] for i, (_b, pre) in enumerate(tloops)
                  if pre is not None}
    claimed: set = set()

    # Fault-site fence: for each block, the index of its last div/mod
    # BinOp.  A hoistable instruction at or before that index stays in
    # the block, since moving it would renumber the faulting BinOp's
    # ordinal within the block (the numbering-independent fault site).
    fence = _fault_fences(new_blocks, block_index)

    for i in process_order:
        if i in candidates:
            _classify_loop(
                i, loops, tloops, process_order, new_blocks, block_index,
                param_defs, candidates, claimed, fence, original_pos,
            )

    # Perform the single physical move per instruction.  Each preheader
    # receives its candidates in the original global instruction order,
    # which SSA dominance guarantees is a valid def-before-use schedule.
    owner = {}
    for block in new_blocks:
        for ins in block.instructions:
            owner[id(ins)] = block

    for i, (_body, pre) in enumerate(tloops):
        cand = candidates.get(i)
        if not cand:
            continue
        for ins in sorted(cand, key=lambda x: original_pos[id(x)]):
            owner[id(ins)].instructions.remove(ins)
            pre.instructions.append(ins)

    return Function(
        func.name,
        [
            Parameter(param.name, param.slot, param.temp)
            for param in func.params
        ],
        func.ret_type,
        list(func.locals),
        new_blocks,
        block_map[func.entry],
        ssa=True,
    )


def _classify_loop(i, loops, tloops, process_order, new_blocks, block_index,
                   param_defs, candidates, claimed, fence,
                   original_pos) -> None:
    """Populate loop ``i``'s candidate set, inner loops first.

    Candidates reach the set in two ways, reconciled to a fixpoint:

    * they are promoted out of a *direct* hoistable child loop's set,
      when their operands are all defined outside this loop or carried
      by another candidate of this same loop;
    * they are claimed directly from the loop's exclusive blocks -- the
      body blocks that no strictly contained loop owns.

    Skipping contained-loop blocks, and promoting only from direct
    hoistable children, makes a loop that fails the preheader gate a
    barrier: a definition inside it can never leave it, even when an
    enclosing loop is hoistable.  Promotion thus proceeds exactly one
    nesting level at a time, mirroring a physical move.

    ``fence`` maps a block's position to the index of its last
    div/mod BinOp; the ``original_pos`` of an eligible instruction must
    fall strictly after it so fault ordinals never shift.
    """
    _header, body_input = loops[i]
    body, _pre = tloops[i]
    cand = candidates[i]

    # Translated body sets of strictly contained loops (gate result
    # irrelevant): their blocks are not this loop's exclusive blocks.
    nested_bodies = [
        tloops[j][0] for j in range(len(loops))
        if loops[j][1] < body_input
    ]
    nested_union = set().union(*nested_bodies) if nested_bodies else set()
    own_blocks = sorted(
        (block for block in body if block not in nested_union),
        key=lambda b: block_index[b],
    )

    outside = set(param_defs)
    for block in new_blocks:
        if block in body:
            continue
        for phi in block.phis:
            outside.add(id(phi.dest))
        for ins in block.instructions:
            outside.add(id(ins.dest))

    # Direct hoistable children: contained loops with no other loop
    # (hoistable or not) strictly between them and this loop.
    def is_direct_child(j) -> bool:
        child_input = loops[j][1]
        if not (child_input < body_input):
            return False
        return not any(
            child_input < loops[k][1] < body_input
            for k in range(len(loops))
        )

    inner_lists = [
        candidates[j] for j in process_order
        if j in candidates and is_direct_child(j)
    ]

    def available(value) -> bool:
        if id(value) in outside:
            return True
        return any(id(ins.dest) == id(value) for ins in cand)

    def eligible(ins, origin=None) -> bool:
        if not _is_hoistable(ins):
            return False
        if not all(available(op) for op in _ins_operands(ins)):
            return False
        # The fault site's ordinal counts BinOps within the block, so
        # only a moved BinOp can shift it; a Const ahead of the trap is
        # free to leave.
        if isinstance(ins, BinOp):
            block_pos, ins_index = (
                original_pos[id(ins)] if origin is None else origin)
            if ins_index <= fence.get(block_pos, -1):
                return False
        return True

    while True:
        progress = False

        # Promote direct-child candidates invariant to this loop.  Their
        # origin is the block/index they were first claimed from, which
        # the fence must keep referring to.
        for inner in inner_lists:
            for ins in list(inner):
                if eligible(ins):
                    inner.remove(ins)
                    cand.append(ins)
                    progress = True

        # Claim still-unclaimed exclusive-block instructions, visiting
        # blocks in block order and instructions in emission order.
        for block in own_blocks:
            block_pos = block_index[block]
            for idx, ins in enumerate(block.instructions):
                if id(ins) in claimed:
                    continue
                if eligible(ins, (block_pos, idx)):
                    claimed.add(id(ins))
                    cand.append(ins)
                    progress = True

        if not progress:
            break


def _fault_fences(new_blocks, block_index):
    """Map block position -> index of its last div/mod BinOp.

    Only arithmetic div/mod BinOps can trap; a comparison or add/sub/mul
    at or before this index is kept so the trap's within-block ordinal
    is unchanged after the instructions ahead of it move.
    """
    fence = {}
    for block in new_blocks:
        last = -1
        for idx, ins in enumerate(block.instructions):
            if (isinstance(ins, BinOp) and ins.kind == "arith"
                    and ins.operator in ("div", "mod")):
                last = idx
        if last >= 0:
            fence[block_index[block]] = last
    return fence


def _is_hoistable(ins) -> bool:
    if isinstance(ins, Const):
        return True
    if isinstance(ins, BinOp):
        if ins.kind == "compare":
            return True
        return ins.kind == "arith" and ins.operator in _HOISTABLE_ARITH
    # Phi never appears among a block's ordinary instructions; Call and
    # Copy are deliberately left in place.
    return False


def _ins_operands(ins):
    if isinstance(ins, BinOp):
        return [ins.left, ins.right]
    if isinstance(ins, Copy):
        return [ins.src]
    if isinstance(ins, Call):
        return list(ins.args)
    return []


# --------------------------------------------------------------------------
# Structural SSA clone (shares only immutable Temps/Slots and literals)
# --------------------------------------------------------------------------


def _clone_into(func: Function, new_blocks: list, block_map: dict) -> None:
    for block, new_block in zip(func.blocks, new_blocks):
        for phi in block.phis:
            entries = {
                block_map[pred]: value
                for pred, value in sorted(
                    phi.entries.items(), key=lambda item: item[0].id
                )
            }
            new_block.phis.append(Phi(phi.dest, entries))

        for ins in block.instructions:
            if isinstance(ins, Const):
                clone = Const(ins.dest, ins.value)
            elif isinstance(ins, Copy):
                clone = Copy(ins.dest, ins.src)
            elif isinstance(ins, BinOp):
                clone = BinOp(
                    ins.dest, ins.operator, ins.left, ins.right,
                    ins.kind, ins.type,
                )
            else:
                clone = Call(ins.dest, ins.name, list(ins.args), ins.type)
            new_block.instructions.append(clone)

        term = block.terminator
        if term is None:
            new_block.terminator = None
        elif isinstance(term, Return):
            new_block.terminator = Return(term.value)
        elif isinstance(term, Jump):
            new_block.terminator = Jump(block_map[term.target])
        elif isinstance(term, Branch):
            new_block.terminator = Branch(
                term.cond,
                block_map[term.true_target],
                block_map[term.false_target],
            )
        else:  # pragma: no cover - lowerer always emits a known terminator
            raise AssertionError(f"unknown terminator: {term!r}")
