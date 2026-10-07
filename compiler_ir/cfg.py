"""Single internal control-flow analysis layer.

The entry point :func:`analyze_function` takes a :class:`~compiler_ir.ir_nodes.Function`
(in either SSA flavor -- the analysis only reads block terminators) and
returns one :class:`FlowAnalysis` bundling every control-flow fact the
compiler needs:

``succs``
    Successor blocks of *every* block of the function, with duplicate edges
    collapsed (a conditional branch whose true and false edge target the
    same block exposes that block exactly once) and ordered by the block's
    position in ``function.blocks``.
``preds``
    Predecessors of every block, counting only edges whose source *and*
    destination are reachable from the entry.  An unreachable block can
    therefore never appear as a reachable block's predecessor, and the
    predecessor lists are ordered by function block position.
``reachable`` / ``reachable_blocks``
    The reachable block set (membership tests) and the same blocks in
    function block order (deterministic traversal).
``rpo``
    Reverse postorder of the reachable subgraph; blocks unreachable from
    the entry never appear.
``idom``
    Immediate dominator of every reachable block.  The entry block
    dominates itself and is its own immediate dominator.
``dom_children``
    Children of every reachable block in the dominator tree, ordered by
    function block position; leaves carry an empty list.
``frontiers``
    Dominance frontier of every reachable block (Cytron et al.), ordered by
    function block position and free of duplicates.

Every result is keyed by the *original* :class:`~compiler_ir.ir_nodes.Block`
objects; nothing here copies or mutates the function.

Determinism
-----------

No observable order derives from set iteration, object identity/address or
hash randomization.  The only orderings ever produced are terminator edge
order (deduplicated), function block position (via ``FlowAnalysis.sort``)
and the explicitly computed reverse postorder.  Immediate dominators use
the Cooper-Harvey-Waterman iterative fix point over that reverse
postorder; the fix point itself is unique, so its result never depends on
the iteration schedule.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ir_nodes import Block, Branch, Function, Jump


@dataclass(frozen=True)
class FlowAnalysis:
    """Immutable bundle of control-flow facts for one function.

    All mappings cover every block of ``function.blocks`` for ``succs`` and
    ``preds``; ``idom``, ``dom_children`` and ``frontiers`` cover exactly the
    reachable blocks.
    """

    blocks: list[Block]
    index: dict[Block, int]
    succs: dict[Block, list[Block]]
    preds: dict[Block, list[Block]]
    reachable: frozenset[Block]
    reachable_blocks: list[Block]
    rpo: list[Block]
    idom: dict[Block, Block]
    dom_children: dict[Block, list[Block]]
    frontiers: dict[Block, list[Block]]

    def sort(self, blocks) -> list[Block]:
        """Order blocks deterministically by function block position."""
        return sorted(blocks, key=lambda b: self.index[b])


def analyze_function(func: Function) -> FlowAnalysis:
    """Compute successors, reachability, RPO, dominators and frontiers.

    The function is read but never modified; the returned analysis is
    keyed by the function's own :class:`Block` objects.
    """
    blocks = list(func.blocks)
    index = {block: i for i, block in enumerate(blocks)}
    succs = {
        block: sorted(_terminator_edges(block), key=index.get)
        for block in blocks
    }

    reachable_set = _reachable_set(func.entry, succs)
    reachable_blocks = [b for b in blocks if b in reachable_set]

    preds: dict[Block, list[Block]] = {block: [] for block in blocks}
    # Sources are visited in function block order, which is also the order
    # appended to each predecessor list; successor lists are already
    # duplicate-free, so every predecessor appears at most once.
    for block in blocks:
        if block not in reachable_set:
            continue
        for target in succs[block]:
            if target in reachable_set:
                preds[target].append(block)

    rpo = _reverse_postorder(succs, reachable_set, func.entry)
    idom = _immediate_dominators(rpo, preds)
    dom_children = _dominator_tree(blocks, reachable_set, idom, index)
    frontiers = _dominance_frontiers(
        reachable_blocks, preds, idom, index
    )

    return FlowAnalysis(
        blocks=blocks,
        index=index,
        succs=succs,
        preds=preds,
        reachable=frozenset(reachable_set),
        reachable_blocks=reachable_blocks,
        rpo=rpo,
        idom=idom,
        dom_children=dom_children,
        frontiers=frontiers,
    )


# --------------------------------------------------------------------------
# CFG primitives
# --------------------------------------------------------------------------


def _terminator_edges(block: Block) -> list[Block]:
    """Return the block's successor blocks, duplicate edges collapsed.

    A :class:`Jump` contributes its target; a :class:`Branch` contributes
    its true and false target in terminator order with a repeated target
    listed once; a :class:`Return` (or a missing terminator) contributes
    nothing.  Callers impose the function-block-position ordering on top:
    terminator order already matches it for IR produced by the lowerer
    (the true edge is always allocated before the false edge).
    """
    term = block.terminator
    if isinstance(term, Jump):
        return [term.target]
    if isinstance(term, Branch):
        if term.true_target is term.false_target:
            return [term.true_target]
        return [term.true_target, term.false_target]
    return []


def _reachable_set(entry: Block, succs: dict[Block, list[Block]]) -> set[Block]:
    """Depth-first set of blocks reachable from ``entry``."""
    seen = {entry}
    stack = [entry]
    while stack:
        block = stack.pop()
        for target in succs[block]:
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def _reverse_postorder(
    succs: dict[Block, list[Block]],
    reachable: frozenset[Block],
    entry: Block,
) -> list[Block]:
    """Iterative DFS reverse postorder over the reachable subgraph.

    Successors are examined in their fixed successor order; the marker is
    taken on push, so join and back edges cannot perturb the traversal.
    """
    order: list[Block] = []
    visited = {entry}
    stack = [(entry, 0)]
    while stack:
        block, pos = stack[-1]
        successors = succs[block]
        if pos < len(successors):
            stack[-1] = (block, pos + 1)
            target = successors[pos]
            if target in reachable and target not in visited:
                visited.add(target)
                stack.append((target, 0))
        else:
            order.append(block)
            stack.pop()
    order.reverse()
    return order


# --------------------------------------------------------------------------
# Dominance (Cooper-Harvey-Waterman) and dominance frontiers (Cytron)
# --------------------------------------------------------------------------


def _immediate_dominators(
    rpo: list[Block], preds: dict[Block, list[Block]]
) -> dict[Block, Block]:
    """Immediate dominators by iterative fix point over reverse postorder.

    The entry's immediate dominator is the entry itself.  Predecessor
    lists already hold only reachable blocks in deterministic order.
    """
    rpo_index = {block: i for i, block in enumerate(rpo)}
    preds_order = {
        block: [p for p in preds[block] if p in rpo_index] for block in rpo
    }

    def intersect(a: Block, b: Block) -> Block:
        while a is not b:
            while rpo_index[a] > rpo_index[b]:
                a = idom[a]
            while rpo_index[b] > rpo_index[a]:
                b = idom[b]
        return a

    idom: dict[Block, Block] = {rpo[0]: rpo[0]}
    entry = rpo[0]
    changed = True
    while changed:
        changed = False
        for block in rpo[1:]:
            new_idom = next(p for p in preds_order[block] if p in idom)
            for pred in preds_order[block]:
                if pred is new_idom or pred not in idom:
                    continue
                new_idom = intersect(pred, new_idom)
            if idom.get(block) is not new_idom:
                idom[block] = new_idom
                changed = True
    assert idom[entry] is entry
    return idom


def _dominator_tree(
    blocks: list[Block],
    reachable: frozenset[Block],
    idom: dict[Block, Block],
    index: dict[Block, int],
) -> dict[Block, list[Block]]:
    """Immediate-dominator children of each reachable block."""
    children: dict[Block, list[Block]] = {block: [] for block in blocks}
    for block in blocks:
        if block not in reachable:
            continue
        if block is not idom[block]:
            children[idom[block]].append(block)
    for child_list in children.values():
        child_list.sort(key=lambda b: index[b])
    # Drop unreachable keys: the tree covers reachable blocks only.
    return {block: children[block] for block in blocks if block in reachable}


def _dominance_frontiers(
    reachable_blocks: list[Block],
    preds: dict[Block, list[Block]],
    idom: dict[Block, Block],
    index: dict[Block, int],
) -> dict[Block, list[Block]]:
    """Dominance frontier of every reachable block.

    A block with a single (deduplicated) predecessor cannot be a control-
    flow merge and contributes nothing; duplicate edges have already been
    merged, so a branch to one and the same target never inflates the
    frontier.
    """
    frontiers: dict[Block, list[Block]] = {
        block: [] for block in reachable_blocks
    }
    sets: dict[Block, set[Block]] = {
        block: set() for block in reachable_blocks
    }
    for block in reachable_blocks:
        if len(preds[block]) < 2:
            continue
        for pred in preds[block]:
            runner = pred
            while runner is not idom[block]:
                sets[runner].add(block)
                runner = idom[runner]
    for block in reachable_blocks:
        frontiers[block] = sorted(sets[block], key=lambda b: index[b])
    return frontiers
