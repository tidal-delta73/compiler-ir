"""Internal control-flow analysis layer.

This module is the single home of the CFG computations shared by SSA
construction and later control-flow-aware optimizations:

* successor edges in fixed terminator order (a branch whose true and false
  edge hit the same block yields one successor, not two);
* predecessors restricted to edges between reachable blocks;
* the reachable block set together with a stable function-order listing;
* a reverse postorder numbering;
* immediate dominators (the entry dominates itself);
* dominator-tree children;
* dominance frontiers.

Every result is keyed by the original :class:`~compiler_ir.ir_nodes.Block`
objects of the analyzed function; the function is never mutated.  Blocks
unreachable from the entry stay in :attr:`ControlFlowAnalysis.blocks` (and
retain their outgoing successors) but never appear in the reverse
postorder, the dominator tree, the dominance frontiers, or in any reachable
block's predecessor list.

All list-valued results use an order derived from function block order or
from fixed terminator edge order -- never from set iteration, object
identity or hashing -- so repeated analyses are observationally identical.

The layer is internal: it is not an optimization pass and is not part of
the package's public exports.
"""
from .ir_nodes import Block, Branch, Function, Jump


class ControlFlowAnalysis:
    """Block-keyed control-flow relations for one :class:`Function`.

    Attributes
    ----------
    blocks:
        The function's blocks in function (label) order.
    index:
        Mapping of every block to its position in :attr:`blocks`.
    succs:
        Every block's successor blocks in fixed terminator edge order
        (jump target, then branch true/false targets), with a repeated
        branch target collapsed to a single entry.
    preds:
        Predecessor blocks, in function block order, counting only edges
        whose source and target are both reachable.  Unreachable blocks
        have an empty list.
    reachable:
        Set of blocks reachable from the entry.
    reachable_order:
        The same blocks as :attr:`reachable`, listed in function order.
    rpo:
        Reachable blocks in reverse postorder of the successor DFS.
    idom:
        Immediate dominator of every reachable block; the entry maps to
        itself.
    children:
        Dominator-tree children of every reachable block, sorted by block
        index.
    frontiers:
        Dominance frontier of every reachable block, sorted by block index.
    """

    __slots__ = (
        "blocks", "index", "succs", "preds", "reachable", "reachable_order",
        "rpo", "idom", "children", "frontiers",
    )

    def __init__(
        self,
        blocks: list[Block],
        index: dict[Block, int],
        succs: dict[Block, list[Block]],
        preds: dict[Block, list[Block]],
        reachable: set[Block],
        reachable_order: list[Block],
        rpo: list[Block],
        idom: dict[Block, Block],
        children: dict[Block, list[Block]],
        frontiers: dict[Block, list[Block]],
    ):
        self.blocks = blocks
        self.index = index
        self.succs = succs
        self.preds = preds
        self.reachable = reachable
        self.reachable_order = reachable_order
        self.rpo = rpo
        self.idom = idom
        self.children = children
        self.frontiers = frontiers

    def sort(self, blocks) -> list[Block]:
        """Sort blocks by their stable function block index."""
        return sorted(blocks, key=lambda b: self.index[b])


def analyze_function(func: Function) -> ControlFlowAnalysis:
    """Compute all control-flow relations of ``func`` without mutating it."""
    blocks = list(func.blocks)
    index = {block: i for i, block in enumerate(blocks)}

    succs: dict[Block, list[Block]] = {
        block: _terminator_edges(block) for block in blocks
    }

    entry = func.entry
    reachable = _reachable_set(succs, entry)
    reachable_order = [b for b in blocks if b in reachable]

    preds: dict[Block, list[Block]] = {block: [] for block in blocks}
    # Walking the blocks in function order keeps every predecessor list in
    # a deterministic order independent of set iteration.  Only edges
    # between reachable blocks count, so an unreachable source never shows
    # up and an edge into an unreachable island is ignored.
    for block in blocks:
        if block not in reachable:
            continue
        for target in succs[block]:
            if target in reachable:
                preds[target].append(block)

    rpo = _reverse_postorder(succs, reachable, entry)
    idom = _immediate_dominators(rpo, preds)
    children = _dominator_children(reachable_order, idom, entry, index)
    frontiers = _dominance_frontiers(reachable_order, preds, idom, index)

    return ControlFlowAnalysis(
        blocks=blocks,
        index=index,
        succs=succs,
        preds=preds,
        reachable=reachable,
        reachable_order=reachable_order,
        rpo=rpo,
        idom=idom,
        children=children,
        frontiers=frontiers,
    )


# --------------------------------------------------------------------------
# Individual relation computations
# --------------------------------------------------------------------------


def _terminator_edges(block: Block) -> list[Block]:
    """Successors of ``block`` in fixed edge order, without duplicates.

    A conditional branch whose true and false targets coincide is one CFG
    edge: the target appears once, so it can never gain a repeated
    predecessor or duplicated dominance information downstream.
    """
    term = block.terminator
    if isinstance(term, Jump):
        return [term.target]
    if isinstance(term, Branch):
        if term.true_target is term.false_target:
            return [term.true_target]
        return [term.true_target, term.false_target]
    return []


def _reachable_set(
    succs: dict[Block, list[Block]], entry: Block
) -> set[Block]:
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
    reachable: set[Block],
    entry: Block,
) -> list[Block]:
    """Reverse postorder of an iterative successor DFS.

    Successors are expanded in fixed edge order, which (together with the
    edge-order deduplication in :func:`_terminator_edges`) makes the
    result independent of hashing.
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


def _immediate_dominators(
    rpo: list[Block], preds: dict[Block, list[Block]]
) -> dict[Block, Block]:
    """Immediate dominators (Cooper-Harvey-Waterman iterative fix point).

    The entry's immediate dominator is the entry itself.  Predecessors are
    consulted in function block order, so the initial candidate and each
    intersection step are free of set-iteration effects.
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
    return idom


def _dominator_children(
    reachable_order: list[Block],
    idom: dict[Block, Block],
    entry: Block,
    index: dict[Block, int],
) -> dict[Block, list[Block]]:
    children: dict[Block, list[Block]] = {
        block: [] for block in reachable_order
    }
    for block in reachable_order:
        if block is not entry:
            children[idom[block]].append(block)
    for child_list in children.values():
        child_list.sort(key=lambda b: index[b])
    return children


def _dominance_frontiers(
    reachable_order: list[Block],
    preds: dict[Block, list[Block]],
    idom: dict[Block, Block],
    index: dict[Block, int],
) -> dict[Block, list[Block]]:
    """Cytron-style dominance frontiers (their Figure 5 loop).

    A runner block can be reached from more than one predecessor of the
    same join (when one predecessor dominates another, its idom chain runs
    through the other's); a per-runner membership guard keeps each frontier
    block unique without relying on a set's iteration order.
    """
    frontiers: dict[Block, list[Block]] = {
        block: [] for block in reachable_order
    }
    for block in reachable_order:
        if len(preds[block]) < 2:
            continue
        for pred in preds[block]:
            runner = pred
            while runner is not idom[block]:
                nodes = frontiers[runner]
                if block not in nodes:
                    nodes.append(block)
                runner = idom[runner]
    for nodes in frontiers.values():
        nodes.sort(key=lambda b: index[b])
    return frontiers
