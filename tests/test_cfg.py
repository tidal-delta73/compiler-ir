"""Tests for the single internal control-flow analysis layer.

The layer (``compiler_ir.cfg.analyze_function``) is exercised on both
hand-built functions -- including shapes ``lower_module`` never emits
(unreachable regions, branches whose two edges hit one block) -- and on
lowered programs.  Besides pinning exact relations on diamonds, nested
branches and back-edge loops, every relation is cross-checked against an
independently computed oracle:

* dominator sets via the classic intersection data-flow equation;
* the immediate dominator as the unique nearest dominator;
* the dominator tree as idom ancestry;
* a dominance-frontier membership test straight from its edge definition
  (``X in DF(B)`` iff some edge ``P -> X`` has ``B`` dominate ``P`` and not
  strictly dominate ``X``);
* reverse postorder as an ordering in which every non-back edge runs
  forward.

All observable orders must be stable lists ordered by function block
position, and the analysis must not mutate its input.
"""
import os
import subprocess
import sys
import unittest

from compiler_ir import (
    Block,
    Branch,
    Const,
    Copy,
    Jump,
    Return,
    Slot,
    Temp,
    Function,
    lower_module,
)
from compiler_ir.cfg import analyze_function

from test_pipeline import (
    arith,
    assign,
    compare,
    func,
    if_,
    int_,
    let,
    param,
    program,
    ret,
    var,
    while_,
)


# --------------------------------------------------------------------------
# Hand-built functions
# --------------------------------------------------------------------------


def _t(n, typ="int"):
    return Temp(n, typ)


def _s(n, typ="int"):
    return Slot(n, typ)


def diamond():
    # b0: cond; br b1 b2 ; b1 -> b3 ; b2 -> b3 ; b3 return
    b0, b1, b2, b3 = (Block(i) for i in range(4))
    slot = _s(0)
    b0.instructions.append(Const(_t(0, "bool"), True))
    b0.terminator = Branch(_t(0, "bool"), b1, b2)
    b1.instructions.append(Const(_t(1), 10))
    b1.instructions.append(Copy(slot, _t(1)))
    b1.terminator = Jump(b3)
    b2.instructions.append(Const(_t(2), 20))
    b2.instructions.append(Copy(slot, _t(2)))
    b2.terminator = Jump(b3)
    b3.terminator = Return(slot)
    return Function("d", [], "int", [slot], [b0, b1, b2, b3], b0)


def nested_branches():
    # b0: br b1 b4
    # b1: br b2 b3
    # b2 -> b5 ; b3 -> b5 ; b4 -> b5
    # b5: return
    b0, b1, b2, b3, b4, b5 = (Block(i) for i in range(6))
    b0.terminator = Branch(_t(0, "bool"), b1, b4)
    b1.terminator = Branch(_t(1, "bool"), b2, b3)
    b2.terminator = Jump(b5)
    b3.terminator = Jump(b5)
    b4.terminator = Jump(b5)
    b5.terminator = Return(None)
    return Function("n", [], "void", [], [b0, b1, b2, b3, b4, b5], b0)


def loop_backedge():
    # b0 -> b1(header); b1: br b2(body) b3(exit); b2 -> b1; b3 return
    blocks = [Block(i) for i in range(4)]
    slot = _s(0)
    blocks[0].instructions.append(Const(_t(0), 0))
    blocks[0].instructions.append(Copy(slot, _t(0)))
    blocks[0].terminator = Jump(blocks[1])
    blocks[1].instructions.append(Const(_t(1, "bool"), True))
    blocks[1].terminator = Branch(_t(1, "bool"), blocks[2], blocks[3])
    blocks[2].instructions.append(Const(_t(2), 5))
    blocks[2].instructions.append(Copy(slot, _t(2)))
    blocks[2].terminator = Jump(blocks[1])
    blocks[3].terminator = Return(slot)
    return Function("l", [], "int", [slot], blocks, blocks[0])


def unreachable_chain():
    # b0 returns; b1 -> b2 -> b3 -> return, all unreachable
    blocks = [Block(i) for i in range(4)]
    blocks[0].instructions.append(Const(_t(0), 1))
    blocks[0].terminator = Return(_t(0))
    blocks[1].terminator = Jump(blocks[2])
    blocks[2].terminator = Jump(blocks[3])
    blocks[3].instructions.append(Const(_t(1), 3))
    blocks[3].terminator = Return(_t(1))
    return Function("u", [], "int", [], blocks, blocks[0])


def unreachable_into_reachable():
    # b0 -> b2 ; b1 (unreachable) -> b2 ; b2 return
    b0, b1, b2 = (Block(i) for i in range(3))
    b0.terminator = Jump(b2)
    b1.instructions.append(Const(_t(0), 9))
    b1.terminator = Jump(b2)
    b2.instructions.append(Const(_t(1), 4))
    b2.terminator = Return(_t(1))
    return Function("u", [], "int", [], [b0, b1, b2], b0)


def dup_target_branch():
    # b0: br b1 b1 ; b1 return
    b0, b1 = Block(0), Block(1)
    b0.instructions.append(Const(_t(0, "bool"), True))
    b0.terminator = Branch(_t(0, "bool"), b1, b1)
    b1.instructions.append(Const(_t(1), 7))
    b1.terminator = Return(_t(1))
    return Function("d", [], "int", [], [b0, b1], b0)


def reversed_edge_order():
    # Allocated b0, b1, b2 but the true edge leaps to the later block:
    # successor order must follow function block position, not edge order.
    b0, b1, b2 = (Block(i) for i in range(3))
    b0.terminator = Branch(_t(0, "bool"), b2, b1)
    b1.terminator = Return(None)
    b2.terminator = Return(None)
    return Function("r", [], "void", [], [b0, b1, b2], b0)


def lowered_counter():
    ast = program(func(
        "counter",
        [param("n", "int")],
        "int",
        [
            let("acc", "int", int_(0)),
            while_(
                compare("gt", var("n"), int_(0)),
                [
                    assign("acc", arith("add", var("acc"), var("n"))),
                    assign("n", arith("sub", var("n"), int_(1))),
                ],
            ),
            ret(var("acc")),
        ],
    ))
    return lower_module(ast).functions[0]


def lowered_diamond():
    ast = program(func(
        "f", [param("c", "bool"), param("x", "int")], "int",
        [
            let("a", "int", int_(1)),
            if_(var("c"), [assign("a", var("x"))], [assign("a", int_(2))]),
            ret(var("a")),
        ],
    ))
    return lower_module(ast).functions[0]


RAW_FUNCTIONS = [
    diamond,
    nested_branches,
    loop_backedge,
    unreachable_chain,
    unreachable_into_reachable,
    dup_target_branch,
    reversed_edge_order,
    lowered_counter,
    lowered_diamond,
]


# --------------------------------------------------------------------------
# Independent oracle
# --------------------------------------------------------------------------


def _raw_edges(block):
    term = block.terminator
    if isinstance(term, Jump):
        return [term.target]
    if isinstance(term, Branch):
        edges = [term.true_target, term.false_target]
        # mirror the analysis' edge dedup for oracle membership
        unique = []
        for target in edges:
            if target not in unique:
                unique.append(target)
        return unique
    return []


def _oracle(func):
    """Reachability, index-ordered pred/succ maps and dominator sets."""
    blocks = list(func.blocks)
    index = {b: i for i, b in enumerate(blocks)}
    succs = {b: sorted(_raw_edges(b), key=index.get) for b in blocks}

    reachable = {func.entry}
    stack = [func.entry]
    while stack:
        b = stack.pop()
        for t in succs[b]:
            if t not in reachable:
                reachable.add(t)
                stack.append(t)

    preds = {b: [] for b in blocks}
    for b in blocks:
        if b not in reachable:
            continue
        for t in succs[b]:
            if t in reachable:
                preds[t].append(b)

    # Classic dominator set equations, iterated from above.
    doms = {func.entry: {func.entry}}
    for b in reachable:
        if b is not func.entry:
            doms[b] = set(reachable)
    changed = True
    while changed:
        changed = False
        for b in sorted(reachable, key=index.get):
            if b is func.entry:
                continue
            incoming = [doms[p] for p in preds[b]]
            new = {b} | (set.intersection(*incoming) if incoming else set())
            if new != doms[b]:
                doms[b] = new
                changed = True
    return blocks, index, succs, preds, reachable, doms


class OracleAssumptions(unittest.TestCase):
    def test_oracle_agrees_with_analysis_on_every_case(self):
        for builder in RAW_FUNCTIONS:
            with self.subTest(builder=builder.__name__):
                func = builder()
                flow = analyze_function(func)
                blocks, index, succs, preds, reachable, doms = _oracle(func)

                # successors / predecessors / reachability
                self.assertEqual(
                    [b.id for b in flow.reachable_blocks],
                    sorted((b.id for b in reachable)),
                )
                self.assertEqual(set(flow.reachable), reachable)
                for b in blocks:
                    self.assertEqual(flow.succs[b], succs[b])
                    self.assertEqual(flow.preds[b], preds[b])

                # immediate dominator: nearest strict dominator (entry self)
                for b in reachable:
                    strict = [d for d in doms[b] if d is not b]
                    expected = b if b is func.entry else next(
                        d for d in strict
                        if not any(
                            d is not e and d in doms[e] for e in strict
                        )
                    )
                    self.assertIs(flow.idom[b], expected)

                # dominator tree: children exactly the blocks this block
                # immediately dominates
                for b in reachable:
                    kids = [
                        x for x in reachable
                        if x is not b and x is not func.entry
                        and flow.idom[x] is b
                    ]
                    self.assertEqual(flow.dom_children[b], sorted(
                        kids, key=index.get
                    ))

                # dominance frontier from its edge definition
                for b in reachable:
                    expected_df = set()
                    for p in reachable:
                        for x in succs[p]:
                            if x not in reachable:
                                continue
                            if b in doms[p] and b not in (
                                doms[x] - {x}
                            ):
                                expected_df.add(x)
                    self.assertEqual(
                        flow.frontiers[b],
                        sorted(expected_df, key=index.get),
                    )


# --------------------------------------------------------------------------
# Exact relations on the named shapes
# --------------------------------------------------------------------------


class DiamondTests(unittest.TestCase):
    def setUp(self):
        self.flow = analyze_function(diamond())

    def test_successors_in_block_order_without_duplicates(self):
        b = self.flow.blocks
        self.assertEqual(self.flow.succs[b[0]], [b[1], b[2]])
        self.assertEqual(self.flow.succs[b[1]], [b[3]])
        self.assertEqual(self.flow.succs[b[2]], [b[3]])
        self.assertEqual(self.flow.succs[b[3]], [])

    def test_predecessors_reachable_only_in_block_order(self):
        b = self.flow.blocks
        self.assertEqual(self.flow.preds[b[0]], [])
        self.assertEqual(self.flow.preds[b[1]], [b[0]])
        self.assertEqual(self.flow.preds[b[2]], [b[0]])
        self.assertEqual(self.flow.preds[b[3]], [b[1], b[2]])

    def test_rpo_starts_at_entry(self):
        b = self.flow.blocks
        self.assertEqual(self.flow.rpo[0], b[0])
        self.assertEqual(set(self.flow.rpo), set(b))

    def test_idom_and_tree(self):
        b = self.flow.blocks
        f = self.flow
        self.assertIs(f.idom[b[0]], b[0])
        self.assertIs(f.idom[b[1]], b[0])
        self.assertIs(f.idom[b[2]], b[0])
        self.assertIs(f.idom[b[3]], b[0])
        self.assertEqual(f.dom_children[b[0]], [b[1], b[2], b[3]])
        self.assertEqual(f.dom_children[b[1]], [])
        self.assertEqual(f.dom_children[b[2]], [])
        self.assertEqual(f.dom_children[b[3]], [])

    def test_frontier_is_the_merge(self):
        b = self.flow.blocks
        f = self.flow
        # The merge b3 lies in the frontier of each arm (they dominate an
        # edge into b3 but do not dominate b3); the entry strictly
        # dominates b3, so b3 is not in its frontier.
        self.assertEqual(f.frontiers[b[0]], [])
        self.assertEqual(f.frontiers[b[1]], [b[3]])
        self.assertEqual(f.frontiers[b[2]], [b[3]])
        self.assertEqual(f.frontiers[b[3]], [])


class NestedBranchTests(unittest.TestCase):
    def setUp(self):
        self.flow = analyze_function(nested_branches())

    def test_idom_chain(self):
        b = self.flow.blocks
        f = self.flow
        self.assertIs(f.idom[b[0]], b[0])
        self.assertIs(f.idom[b[1]], b[0])
        self.assertIs(f.idom[b[2]], b[1])
        self.assertIs(f.idom[b[3]], b[1])
        # b4 comes straight off the outer branch; b5 joins all three arms.
        self.assertIs(f.idom[b[4]], b[0])
        self.assertIs(f.idom[b[5]], b[0])

    def test_nested_frontiers(self):
        b = self.flow.blocks
        f = self.flow
        # The outer merge b5 is in the frontier of every arm block
        # leading into it (b2/b3 under the inner branch, b4 directly) and
        # of the inner branch b1; b0 strictly dominates b5.
        self.assertEqual(f.frontiers[b[0]], [])
        self.assertEqual(f.frontiers[b[1]], [b[5]])
        self.assertEqual(f.frontiers[b[2]], [b[5]])
        self.assertEqual(f.frontiers[b[3]], [b[5]])
        self.assertEqual(f.frontiers[b[4]], [b[5]])
        self.assertEqual(f.frontiers[b[5]], [])

    def test_tree_children_sorted(self):
        b = self.flow.blocks
        self.assertEqual(self.flow.dom_children[b[0]], [b[1], b[4], b[5]])
        self.assertEqual(self.flow.dom_children[b[1]], [b[2], b[3]])


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.flow = analyze_function(loop_backedge())

    def test_rpo_header_precedes_exit_and_body_finishes_last(self):
        b = self.flow.blocks
        # DFS takes the body edge first and meets the back edge there, so
        # the exit is finished before the body; the header is second.
        self.assertEqual(self.flow.rpo, [b[0], b[1], b[3], b[2]])

    def test_idom_through_back_edge(self):
        b = self.flow.blocks
        f = self.flow
        self.assertIs(f.idom[b[0]], b[0])
        self.assertIs(f.idom[b[1]], b[0])
        self.assertIs(f.idom[b[2]], b[1])
        self.assertIs(f.idom[b[3]], b[1])

    def test_back_edge_marks_header_frontier(self):
        b = self.flow.blocks
        f = self.flow
        # The body runs to the header without being dominated by it on the
        # entry edge, so both the body and the header list the header.
        self.assertEqual(f.frontiers[b[0]], [])
        self.assertEqual(f.frontiers[b[1]], [b[1]])
        self.assertEqual(f.frontiers[b[2]], [b[1]])
        self.assertEqual(f.frontiers[b[3]], [])

    def test_lowered_counter_has_same_shape(self):
        f = analyze_function(lowered_counter())
        b = f.blocks
        self.assertEqual(len(b), 4)
        self.assertEqual(f.rpo, [b[0], b[1], b[3], b[2]])
        self.assertIs(f.idom[b[1]], b[0])
        self.assertIs(f.idom[b[2]], b[1])
        self.assertIs(f.idom[b[3]], b[1])
        self.assertEqual(f.frontiers[b[2]], [b[1]])


class UnreachableTests(unittest.TestCase):
    def test_unreachable_chain_excluded_everywhere_but_block_list(self):
        func = unreachable_chain()
        f = analyze_function(func)
        b = func.blocks
        # All blocks are still indexed ...
        self.assertEqual(f.blocks, b)
        self.assertEqual([f.succs[x] for x in b], [[], [b[2]], [b[3]], []])
        # ... but only the entry is reachable.
        self.assertEqual(f.reachable_blocks, [b[0]])
        self.assertEqual(set(f.reachable), {b[0]})
        self.assertEqual(f.rpo, [b[0]])
        self.assertEqual(list(f.idom), [b[0]])
        self.assertIs(f.idom[b[0]], b[0])
        self.assertEqual(list(f.dom_children), [b[0]])
        self.assertEqual(f.dom_children[b[0]], [])
        self.assertEqual(list(f.frontiers), [b[0]])
        self.assertEqual(f.frontiers[b[0]], [])
        # Unreachable sources never populate predecessor lists.
        for block in b:
            self.assertEqual(f.preds[block], [])

    def test_unreachable_predator_of_reachable_block_is_hidden(self):
        func = unreachable_into_reachable()
        f = analyze_function(func)
        b = func.blocks
        self.assertEqual(f.reachable_blocks, [b[0], b[2]])
        self.assertEqual(f.rpo, [b[0], b[2]])
        # The reachable merge block sees only its reachable predecessor.
        self.assertEqual(f.preds[b[2]], [b[0]])
        self.assertNotIn(b[1], f.preds[b[2]])
        self.assertIs(f.idom[b[2]], b[0])
        self.assertEqual(f.frontiers[b[0]], [])
        self.assertEqual(f.frontiers[b[2]], [])

    def test_unreachable_blocks_keep_function_block_sequence(self):
        func = unreachable_chain()
        self.assertEqual(
            [x.id for x in func.blocks],
            [x.id for x in analyze_function(func).blocks],
        )


class DuplicateTargetTests(unittest.TestCase):
    def test_branch_with_two_equal_targets_has_no_duplicate_edges(self):
        func = dup_target_branch()
        f = analyze_function(func)
        b = func.blocks
        self.assertEqual(f.succs[b[0]], [b[1]])
        self.assertEqual(f.preds[b[1]], [b[0]])
        self.assertEqual(f.rpo, [b[0], b[1]])
        self.assertIs(f.idom[b[1]], b[0])
        self.assertEqual(f.dom_children[b[0]], [b[1]])
        # A one-predecessor target is no merge: empty frontiers all round.
        self.assertEqual(f.frontiers[b[0]], [])
        self.assertEqual(f.frontiers[b[1]], [])

    def test_successors_follow_function_position_not_edge_order(self):
        func = reversed_edge_order()
        f = analyze_function(func)
        b = func.blocks
        # True edge points at b2 but the stable order is b1 then b2.
        self.assertEqual(f.succs[b[0]], [b[1], b[2]])
        self.assertEqual(f.rpo[0], b[0])
        self.assertEqual(set(f.rpo), {b[0], b[1], b[2]})


# --------------------------------------------------------------------------
# Ordering discipline
# --------------------------------------------------------------------------


class OrderingTests(unittest.TestCase):
    def test_every_list_is_index_ordered(self):
        for builder in RAW_FUNCTIONS:
            with self.subTest(builder=builder.__name__):
                func = builder()
                f = analyze_function(func)
                index = f.index

                def ordered(seq):
                    ids = [index[x] for x in seq]
                    return ids == sorted(ids) and len(ids) == len(set(ids))

                for seq in f.succs.values():
                    self.assertTrue(ordered(seq))
                for seq in f.preds.values():
                    self.assertTrue(ordered(seq))
                self.assertTrue(ordered(f.reachable_blocks))
                for seq in f.dom_children.values():
                    self.assertTrue(ordered(seq))
                for seq in f.frontiers.values():
                    self.assertTrue(ordered(seq))

    def test_rpo_respects_every_non_back_edge(self):
        # An edge u -> v is a back edge exactly when v dominates u; every
        # other edge must run forward in reverse postorder.
        for builder in RAW_FUNCTIONS:
            with self.subTest(builder=builder.__name__):
                func = builder()
                f = analyze_function(func)
                _, _, _, _, _, doms = _oracle(func)
                pos = {b: i for i, b in enumerate(f.rpo)}
                for u in f.reachable:
                    for v in f.succs[u]:
                        if v not in f.reachable:
                            continue
                        if v in doms[u]:  # v dominates u -> back edge
                            continue
                        self.assertLess(pos[u], pos[v])

    def test_repeated_analysis_is_identical(self):
        for builder in RAW_FUNCTIONS:
            first = analyze_function(builder())
            second = analyze_function(builder())

            def signature(flow):
                return (
                    [flow.index[b] for b in flow.rpo],
                    [(flow.index[k], [flow.index[x] for x in v])
                     for k, v in flow.succs.items()],
                    [(flow.index[k], [flow.index[x] for x in v])
                     for k, v in flow.preds.items()],
                    sorted(
                        (flow.index[k], flow.index[v])
                        for k, v in flow.idom.items()
                    ),
                    sorted(
                        (flow.index[k], [flow.index[x] for x in v])
                        for k, v in flow.frontiers.items()
                    ),
                )

            self.assertEqual(signature(first), signature(second))

    def test_order_independent_of_hash_seed(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        snippet = (
            "import sys; sys.path[:0] = [%r, %r];"
            "from tests.test_cfg import signature_for_subprocess;"
            "print(signature_for_subprocess())" % (repo, tests_dir)
        )
        env = dict(os.environ, PYTHONPATH=repo)
        outputs = []
        for seed in ("0", "1", "42", "123456"):
            proc = subprocess.run(
                [sys.executable, "-c", snippet],
                env=dict(env, PYTHONHASHSEED=seed),
                capture_output=True,
                text=True,
                check=True,
            )
            outputs.append(proc.stdout)
        self.assertEqual(len(set(outputs)), 1, outputs)


def signature_for_subprocess():
    """Deterministic textual signature over all shapes (hash-seed probe)."""
    parts = []
    for builder in RAW_FUNCTIONS:
        f = analyze_function(builder())
        parts.append(repr((
            [f.index[b] for b in f.rpo],
            sorted(
                (f.index[k], [f.index[x] for x in v])
                for k, v in f.succs.items()
            ),
            sorted(
                (f.index[k], [f.index[x] for x in v])
                for k, v in f.preds.items()
            ),
            sorted(
                (f.index[k], f.index[v]) for k, v in f.idom.items()
            ),
            sorted(
                (f.index[k], [f.index[x] for x in v])
                for k, v in f.frontiers.items()
            ),
        )))
    return "|".join(parts)


# --------------------------------------------------------------------------
# Purity
# --------------------------------------------------------------------------


class PurityTests(unittest.TestCase):
    def test_analysis_does_not_mutate_function(self):
        from compiler_ir import Module, render_module

        func = lowered_diamond()
        module = Module([func])
        before = render_module(module)
        block_attrs = [(b.id, len(b.instructions), type(b.terminator))
                       for b in func.blocks]
        analyze_function(func)
        analyze_function(func)
        self.assertEqual(render_module(module), before)
        self.assertEqual(
            [(b.id, len(b.instructions), type(b.terminator))
             for b in func.blocks],
            block_attrs,
        )


if __name__ == "__main__":
    unittest.main()
