"""Tests for the internal control-flow analysis layer.

``analyze_function`` is internal (not exported from :mod:`compiler_ir`); it
is the single reusable home of reachability, predecessors, reverse
postorder, immediate dominators, dominator-tree children and dominance
frontiers.  The cases pin those relations down on diamond branches, nested
branches, loops with back edges, unreachable blocks and branches whose two
targets coincide, and additionally require:

* results keyed by the *original* block objects;
* successors in fixed terminator order, predecessors in function block
  order, with no duplicated edges;
* unreachable blocks retained in the function block sequence but excluded
  from the reachable set, the reverse postorder, the dominator tree and the
  frontiers, and never appearing as a reachable block's predecessor;
* the entry block dominating itself;
* ordering independent of set iteration and hash randomization;
* no mutation of the analyzed function.
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
    Module,
    Parameter,
    lower_module,
    render_module,
    to_ssa,
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
# Helpers
# --------------------------------------------------------------------------


def ids(blocks):
    return [b.id for b in blocks]


def relation_ids(relation, order):
    return [(b.id, ids(relation[b])) for b in order]


def diamond_module():
    # 0 -> 1,2 ; 1 -> 3 ; 2 -> 3 ; 3 returns
    return lower_module(program(func(
        "diamond", [param("c", "bool"), param("x", "int")], "int",
        [
            let("a", "int", int_(1)),
            if_(var("c"), [assign("a", var("x"))], [assign("a", int_(2))]),
            ret(var("a")),
        ],
    )))


def nested_module():
    return lower_module(program(func(
        "nested",
        [param("c", "bool"), param("d", "bool"), param("x", "int")],
        "int",
        [
            let("a", "int", int_(0)),
            if_(var("c"),
                [if_(var("d"),
                     [assign("a", var("x"))],
                     [assign("a", int_(9))])],
                [assign("a", int_(7))]),
            ret(var("a")),
        ],
    )))


def loop_module():
    return lower_module(program(func(
        "counter", [param("n", "int")], "int",
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
    )))


# --------------------------------------------------------------------------
# Relation assertions
# --------------------------------------------------------------------------


class DiamondAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fn = diamond_module().functions[0]
        self.a = analyze_function(self.fn)

    def test_results_keyed_by_original_blocks(self):
        self.assertEqual(self.a.blocks, list(self.fn.blocks))
        self.assertIs(self.a.rpo[0], self.fn.entry)
        for block in self.fn.blocks:
            self.assertIn(block, self.a.succs)
            self.assertIn(block, self.a.preds)
            self.assertIn(block, self.a.index)
        for block in self.a.reachable:
            self.assertIn(block, self.a.idom)
            self.assertIn(block, self.a.children)
            self.assertIn(block, self.a.frontiers)

    def test_successors_in_fixed_edge_order(self):
        b = self.a.blocks
        self.assertEqual(ids(self.a.succs[b[0]]), [1, 2])
        self.assertEqual(ids(self.a.succs[b[1]]), [3])
        self.assertEqual(ids(self.a.succs[b[2]]), [3])
        self.assertEqual(self.a.succs[b[3]], [])

    def test_predecessors_in_function_block_order(self):
        b = self.a.blocks
        self.assertEqual(ids(self.a.preds[b[0]]), [])
        self.assertEqual(ids(self.a.preds[b[1]]), [0])
        self.assertEqual(ids(self.a.preds[b[2]]), [0])
        self.assertEqual(ids(self.a.preds[b[3]]), [1, 2])

    def test_reachable_and_reverse_postorder(self):
        self.assertEqual(ids(self.a.reachable_order), [0, 1, 2, 3])
        self.assertEqual(set(self.a.reachable), set(self.a.blocks))
        self.assertEqual(ids(self.a.rpo), [0, 2, 1, 3])

    def test_immediate_dominators_entry_is_itself(self):
        b = self.a.blocks
        self.assertIs(self.a.idom[b[0]], b[0])
        self.assertIs(self.a.idom[b[1]], b[0])
        self.assertIs(self.a.idom[b[2]], b[0])
        self.assertIs(self.a.idom[b[3]], b[0])

    def test_dominator_children_in_block_order(self):
        b = self.a.blocks
        self.assertEqual(ids(self.a.children[b[0]]), [1, 2, 3])
        self.assertEqual(self.a.children[b[1]], [])
        self.assertEqual(self.a.children[b[2]], [])
        self.assertEqual(self.a.children[b[3]], [])

    def test_dominance_frontiers(self):
        b = self.a.blocks
        self.assertEqual(ids(self.a.frontiers[b[0]]), [])
        self.assertEqual(ids(self.a.frontiers[b[1]]), [3])
        self.assertEqual(ids(self.a.frontiers[b[2]]), [3])
        self.assertEqual(ids(self.a.frontiers[b[3]]), [])


class NestedBranchAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fn = nested_module().functions[0]
        self.a = analyze_function(self.fn)

    def test_relations(self):
        b = self.a.blocks
        # 0->1,2 ; 1->3,4 ; 2->6 ; 3->5 ; 4->5 ; 5->6
        self.assertEqual(ids(self.a.succs[b[0]]), [1, 2])
        self.assertEqual(ids(self.a.succs[b[1]]), [3, 4])
        self.assertEqual(ids(self.a.succs[b[2]]), [6])
        self.assertEqual(ids(self.a.succs[b[3]]), [5])
        self.assertEqual(ids(self.a.succs[b[4]]), [5])
        self.assertEqual(ids(self.a.succs[b[5]]), [6])
        self.assertEqual(ids(self.a.rpo), [0, 2, 1, 4, 3, 5, 6])

        self.assertIs(self.a.idom[b[0]], b[0])
        self.assertIs(self.a.idom[b[1]], b[0])
        self.assertIs(self.a.idom[b[2]], b[0])
        self.assertIs(self.a.idom[b[3]], b[1])
        self.assertIs(self.a.idom[b[4]], b[1])
        self.assertIs(self.a.idom[b[5]], b[1])
        self.assertIs(self.a.idom[b[6]], b[0])

        self.assertEqual(ids(self.a.children[b[0]]), [1, 2, 6])
        self.assertEqual(ids(self.a.children[b[1]]), [3, 4, 5])

        self.assertEqual(ids(self.a.frontiers[b[1]]), [6])
        self.assertEqual(ids(self.a.frontiers[b[2]]), [6])
        self.assertEqual(ids(self.a.frontiers[b[3]]), [5])
        self.assertEqual(ids(self.a.frontiers[b[4]]), [5])
        self.assertEqual(ids(self.a.frontiers[b[5]]), [6])

    def test_every_dom_child_appears_exactly_once(self):
        seen = []
        for child_list in self.a.children.values():
            seen.extend(child_list)
        # All reachable blocks except the entry appear exactly once.
        self.assertEqual(ids(sorted(seen, key=lambda x: x.id)),
                         [1, 2, 3, 4, 5, 6])


class LoopAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fn = loop_module().functions[0]
        self.a = analyze_function(self.fn)

    def test_back_edge_kept_in_successors_and_predecessors(self):
        b = self.a.blocks
        # 0->1 ; 1->2,3 ; 2->1 (back edge) ; 3 returns
        self.assertEqual(ids(self.a.succs[b[2]]), [1])
        self.assertEqual(ids(self.a.preds[b[1]]), [0, 2])

    def test_rpo_resolves_back_edge(self):
        self.assertEqual(ids(self.a.rpo), [0, 1, 3, 2])

    def test_loop_header_self_dominance_frontier(self):
        b = self.a.blocks
        self.assertIs(self.a.idom[b[1]], b[0])
        self.assertIs(self.a.idom[b[2]], b[1])
        self.assertIs(self.a.idom[b[3]], b[1])
        # The header (b1) is in its own frontier via the back edge.
        self.assertEqual(ids(self.a.frontiers[b[1]]), [1])
        self.assertEqual(ids(self.a.frontiers[b[2]]), [1])
        self.assertEqual(ids(self.a.children[b[1]]), [2, 3])


# --------------------------------------------------------------------------
# Hand-built edge cases (shapes lower_module never emits)
# --------------------------------------------------------------------------


def unreachable_function():
    # b0 returns directly (reachable: b0 only).
    # b1 -> b2 (unreachable island).
    # b2 unreachable, jumps back to the reachable block b0.
    # b3 unreachable self-loop branch: br -> b3, b3.
    b0 = Block(0)
    b1 = Block(1)
    b2 = Block(2)
    b3 = Block(3)
    t0 = Temp(0, "int")
    t1 = Temp(1, "int")
    t2 = Temp(2, "bool")
    v = Slot(0, "int")
    b0.instructions = [Const(t0, 5)]
    b0.terminator = Return(t0)
    b1.instructions = [Const(t1, 6), Copy(v, t1)]
    b1.terminator = Jump(b2)
    b2.terminator = Jump(b0)
    b3.instructions = [Const(t2, True)]
    b3.terminator = Branch(t2, b3, b3)
    return Function(
        "unreach",
        [Parameter("x", v)],
        "int",
        [v],
        [b0, b1, b2, b3],
        b0,
    )


def dup_target_function():
    # b0 branches with identical true/false target b1.
    b0 = Block(0)
    b1 = Block(1)
    v = Slot(0, "int")
    cond = Temp(0, "bool")
    b0.instructions = [Const(cond, True)]
    b0.terminator = Branch(cond, b1, b1)
    b1.terminator = Return(v)
    return Function(
        "dup", [Parameter("x", v)], "int", [], [b0, b1], b0
    )


def cascade_function():
    # Three converging paths where one join predecessor dominates another:
    # b0 -> b1, b3 ; b1 -> b2, b3 ; b2 -> b3 ; b3 returns.
    b0 = Block(0)
    b1 = Block(1)
    b2 = Block(2)
    b3 = Block(3)
    t = Temp(0, "int")
    c0 = Temp(1, "bool")
    c1 = Temp(2, "bool")
    b0.instructions = [Const(c0, True)]
    b0.terminator = Branch(c0, b1, b3)
    b1.instructions = [Const(c1, False)]
    b1.terminator = Branch(c1, b2, b3)
    b2.terminator = Jump(b3)
    b3.terminator = Return(t)
    return Function("cascade", [], "int", [], [b0, b1, b2, b3], b0)


class UnreachableBlockTests(unittest.TestCase):
    def setUp(self):
        self.fn = unreachable_function()
        self.a = analyze_function(self.fn)

    def test_unreachable_blocks_kept_in_block_sequence(self):
        self.assertEqual(ids(self.a.blocks), [0, 1, 2, 3])

    def test_only_entry_reachable(self):
        self.assertEqual(ids(self.a.reachable_order), [0])
        self.assertEqual(set(self.a.reachable), {self.fn.blocks[0]})

    def test_rpo_dom_tree_and_frontiers_exclude_unreachable(self):
        b = self.a.blocks
        self.assertEqual(ids(self.a.rpo), [0])
        self.assertEqual(list(self.a.idom), [b[0]])
        self.assertIs(self.a.idom[b[0]], b[0])
        self.assertEqual(list(self.a.children), [b[0]])
        self.assertEqual(self.a.children[b[0]], [])
        self.assertEqual(list(self.a.frontiers), [b[0]])
        self.assertEqual(self.a.frontiers[b[0]], [])

    def test_unreachable_block_is_never_a_reachable_predecessor(self):
        b = self.a.blocks
        # b2 jumps to b0, but b2 is unreachable: b0 must have no preds.
        self.assertEqual(self.a.preds[b[0]], [])
        # Unreachable blocks carry no reachable-only predecessors either.
        self.assertEqual(self.a.preds[b[1]], [])
        self.assertEqual(self.a.preds[b[2]], [])
        self.assertEqual(self.a.preds[b[3]], [])

    def test_unreachable_successors_are_still_listed(self):
        b = self.a.blocks
        self.assertEqual(ids(self.a.succs[b[1]]), [2])
        self.assertEqual(ids(self.a.succs[b[2]]), [0])
        self.assertEqual(ids(self.a.succs[b[3]]), [3])

    def test_ssa_conversion_keeps_unreachable_block_structured(self):
        # The block survives in order, with no phis and no reachable preds.
        result = to_ssa(Module([self.fn])).functions[0]
        self.assertEqual([blk.id for blk in result.blocks], [0, 1, 2, 3])
        self.assertTrue(all(blk.phis == [] for blk in result.blocks))


class DuplicateTargetBranchTests(unittest.TestCase):
    def setUp(self):
        self.fn = dup_target_function()
        self.a = analyze_function(self.fn)

    def test_single_successor_for_coincident_edges(self):
        self.assertEqual(ids(self.a.succs[self.fn.blocks[0]]), [1])

    def test_single_predecessor_no_duplicate(self):
        self.assertEqual(ids(self.a.preds[self.fn.blocks[1]]), [0])

    def test_no_duplicate_dominance_information(self):
        b = self.fn.blocks
        self.assertIs(self.a.idom[b[1]], b[0])
        self.assertEqual(ids(self.a.children[b[0]]), [1])
        self.assertEqual(self.a.frontiers[b[0]], [])
        self.assertEqual(self.a.frontiers[b[1]], [])
        # Block b1 has just one predecessor, so no join frontier exists.
        self.assertEqual(len(self.a.preds[b[1]]), 1)

    def test_ssa_output_has_no_phi_for_single_edge(self):
        result = to_ssa(Module([self.fn])).functions[0]
        self.assertFalse(
            any(blk.phis for blk in result.blocks)
        )
        merge = result.blocks[1]
        self.assertIs(merge.terminator.value, result.params[0].temp)


class CascadeJoinTests(unittest.TestCase):
    def setUp(self):
        self.fn = cascade_function()
        self.a = analyze_function(self.fn)

    def test_dominated_predecessor_adds_no_duplicate_frontier(self):
        b = self.fn.blocks
        # b3 has three predecessors, one of which (b0) dominates the others
        # and is b3's immediate dominator.
        self.assertEqual(ids(self.a.preds[b[3]]), [0, 1, 2])
        self.assertIs(self.a.idom[b[3]], b[0])
        # Walking up from b2 reaches b1 a second time: b3 must still appear
        # exactly once in b1's frontier.
        self.assertEqual(ids(self.a.frontiers[b[1]]), [3])
        self.assertEqual(ids(self.a.frontiers[b[2]]), [3])
        self.assertEqual(ids(self.a.frontiers[b[0]]), [])

    def test_all_frontier_lists_have_unique_entries(self):
        for block, nodes in self.a.frontiers.items():
            self.assertEqual(len(nodes), len(set(nodes)), block.label)


# --------------------------------------------------------------------------
# Determinism and purity
# --------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    PROGRAMS = [diamond_module(), nested_module(), loop_module(),
                Module([unreachable_function()]),
                Module([dup_target_function()]),
                Module([cascade_function()])]

    def _signature(self, a):
        return repr((
            relation_ids(a.succs, a.blocks),
            relation_ids(a.preds, a.blocks),
            ids(a.reachable_order),
            ids(a.rpo),
            [(b.id, a.idom[b].id) for b in a.reachable_order],
            relation_ids(a.children, a.reachable_order),
            relation_ids(a.frontiers, a.reachable_order),
        ))

    def test_repeated_analysis_is_identical(self):
        for module in self.PROGRAMS:
            fn = module.functions[0]
            first = self._signature(analyze_function(fn))
            for _ in range(5):
                self.assertEqual(first, self._signature(analyze_function(fn)))

    def test_repeated_analysis_returns_fresh_containers(self):
        fn = self.PROGRAMS[0].functions[0]
        a1 = analyze_function(fn)
        a2 = analyze_function(fn)
        self.assertIsNot(a1.succs, a2.succs)
        self.assertIsNot(a1.preds, a2.preds)
        self.assertIsNot(a1.rpo, a2.rpo)
        self.assertIsNot(a1.frontiers, a2.frontiers)
        self.assertIsNot(a1.succs[fn.blocks[0]], a2.succs[fn.blocks[0]])

    def test_order_independent_of_hash_seed(self):
        # Run the relation dump in fresh interpreters with deliberately
        # different hash seeds: any dependence on set/object hashing would
        # perturb ordering.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = (
            "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r);\n"
            "from compiler_ir.cfg import analyze_function\n"
            "from test_cfg import DeterminismTests\n"
            "d = DeterminismTests()\n"
            "mods = d.PROGRAMS\n"
            "out = []\n"
            "for m in mods:\n"
            "    out.append(d._signature(analyze_function(m.functions[0])))\n"
            "print('\\n'.join(out))\n"
        ) % (repo_root, os.path.join(repo_root, "tests"))
        outputs = []
        for seed in ("0", "1", "12345", "999999"):
            env = dict(os.environ, PYTHONPATH=repo_root, PYTHONHASHSEED=seed)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env, check=True,
            )
            outputs.append(proc.stdout)
        self.assertEqual(len(set(outputs)), 1)

    def test_analysis_does_not_mutate_function(self):
        for module in self.PROGRAMS:
            fn = module.functions[0]
            before = render_module(module)
            analyze_function(fn)
            analyze_function(fn)
            self.assertEqual(render_module(module), before)

    def test_analysis_works_on_ssa_function_with_same_relations(self):
        non_ssa = diamond_module().functions[0]
        ssa = to_ssa(Module([non_ssa])).functions[0]
        a_before = analyze_function(non_ssa)
        a_after = analyze_function(ssa)
        self.assertEqual(ids(a_before.rpo), ids(a_after.rpo))
        self.assertEqual(
            relation_ids(a_before.frontiers, a_before.reachable_order),
            relation_ids(a_after.frontiers, a_after.reachable_order),
        )
        self.assertEqual(
            [(b.id, a_before.idom[b].id)
             for b in a_before.reachable_order],
            [(b.id, a_after.idom[b].id)
             for b in a_after.reachable_order],
        )
        self.assertEqual(
            relation_ids(a_before.children, a_before.reachable_order),
            relation_ids(a_after.children, a_after.reachable_order),
        )


if __name__ == "__main__":
    unittest.main()
