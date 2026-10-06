"""Tests for the non-SSA -> pruned SSA conversion (:func:`to_ssa`).

The tests pin down: unique definitions and slot elimination; phi placement
at real joins (if/else, missing else, nesting, loop headers, short-circuit
merges); phi ordering and reachable-only predecessors; redundant-phi
pruning; preservation of calls, arithmetic, comparisons, branches and
return values; deterministic per-function numbering; byte-identical
idempotence; ``TypeError`` for non-modules; and full compatibility of the
existing non-SSA pipeline.
"""
import unittest

from compiler_ir import (
    BinOp,
    Branch,
    Call,
    Copy,
    Jump,
    Module,
    Phi,
    Return,
    Slot,
    Temp,
    emit_ir,
    lower_module,
    render_module,
    to_ssa,
)

from test_pipeline import (
    arith,
    assign,
    bool_,
    call,
    compare,
    func,
    if_,
    int_,
    let,
    logical,
    param,
    program,
    ret,
    var,
    while_,
)


# --------------------------------------------------------------------------
# Structural helpers
# --------------------------------------------------------------------------


def _edges(block):
    term = block.terminator
    if isinstance(term, Jump):
        return [term.target]
    elif isinstance(term, Branch):
        return [term.true_target, term.false_target]
    return []


def _operands(ins):
    if isinstance(ins, Copy):
        return [ins.src]
    if isinstance(ins, BinOp):
        return [ins.left, ins.right]
    if isinstance(ins, Call):
        return list(ins.args)
    return []


def _reachable(func):
    seen = {func.entry}
    stack = [func.entry]
    while stack:
        block = stack.pop()
        for target in _edges(block):
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def _predecessors(func):
    reach = _reachable(func)
    preds = {block: [] for block in func.blocks}
    for block in func.blocks:
        if block in reach:
            for target in _edges(block):
                if target in reach:
                    preds[target].append(block)
    return preds, reach


def _all_values(func):
    """Every operand value appearing anywhere in a function."""
    values = []
    for block in func.blocks:
        for phi in block.phis:
            values.extend(phi.entries.values())
        for ins in block.instructions:
            values.extend(_operands(ins))
        term = block.terminator
        if isinstance(term, Return):
            if term.value is not None:
                values.append(term.value)
        elif isinstance(term, Branch):
            values.append(term.cond)
    return values


def _defined_temps(func):
    """Every definition, in canonical order (params, phis, instructions)."""
    defs = [p.temp for p in func.params]
    for block in func.blocks:
        defs.extend(phi.dest for phi in block.phis)
        defs.extend(ins.dest for ins in block.instructions)
    return defs


def assert_ssa_well_formed(test, func):
    preds, reach = _predecessors(func)

    # Every definition is a Temp with a unique, dense per-function id.
    defs = _defined_temps(func)
    ids = [d.id for d in defs]
    test.assertEqual(sorted(ids), list(range(len(ids))))
    test.assertFalse(any(isinstance(d, Slot) for d in defs))

    defblock = {}
    for param in func.params:
        defblock[id(param.temp)] = func.entry
    for blk in func.blocks:
        for phi in blk.phis:
            defblock[id(phi.dest)] = blk
        for ins in blk.instructions:
            defblock[id(ins.dest)] = blk

    # Every use is a defined Temp (no Slots), and phis sit before ordinary
    # instructions with exactly the reachable predecessors, in label order.
    for blk in func.blocks:
        if blk not in reach:
            continue
        for phi in blk.phis:
            keys = list(phi.entries.keys())
            test.assertEqual(
                [k.id for k in keys], sorted(k.id for k in keys)
            )
            test.assertEqual(set(keys), set(preds[blk]))
            for value in phi.entries.values():
                test.assertIsInstance(value, Temp)
                test.assertIn(id(value), defblock)
        for value in _all_values(func):
            test.assertIsInstance(value, Temp)
            test.assertIn(id(value), defblock)
        test.assertEqual(
            [type(i) for i in blk.instructions].count(Phi), 0
        )


# --------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------


class ApiTests(unittest.TestCase):
    def test_returns_new_module_marked_ssa(self):
        module = lower_module(program(COUNTER))
        result = to_ssa(module)
        self.assertIsInstance(result, Module)
        self.assertIsNot(result, module)
        self.assertTrue(result.ssa)
        for fn_ in result.functions:
            self.assertTrue(fn_.ssa)

    def test_input_module_is_not_mutated(self):
        module = lower_module(program(COUNTER))
        before = render_module(module)
        to_ssa(module)
        self.assertEqual(render_module(module), before)
        # Original still reports non-SSA and has no phi nodes.
        self.assertFalse(module.ssa)
        self.assertFalse(any(
            block.phis for fn_ in module.functions
            for block in fn_.blocks
        ))
        self.assertIsNone(module.functions[0].params[0].temp)

    def test_non_module_raises_type_error(self):
        for bad in (None, 1, "module", [], {}, object()):
            with self.assertRaises(TypeError):
                to_ssa(bad)

    def test_parameters_become_numbered_temps(self):
        module = lower_module(
            program(func("add", [param("a", "int"), param("b", "int")],
                        "int", [ret(arith("add", var("a"), var("b")))]))
        )
        fn = to_ssa(module).functions[0]
        self.assertEqual([p.temp.id for p in fn.params], [0, 1])
        self.assertEqual([p.temp.type for p in fn.params], ["int", "int"])
        # No locals and no slots survive in SSA form.
        self.assertEqual(fn.locals, [])
        self.assertFalse(any(
            isinstance(v, Slot) for v in _all_values(fn)
        ))


# --------------------------------------------------------------------------
# Phi placement
# --------------------------------------------------------------------------


def join_if_program():
    return program(func(
        "f", [param("c", "bool"), param("x", "int")], "int",
        [
            let("a", "int", int_(1)),
            if_(var("c"), [assign("a", var("x"))],
                [assign("a", int_(2))]),
            ret(var("a")),
        ],
    ))


# Reuse the assignment helper from test_pipeline's namespace.


class PhiPlacementTests(unittest.TestCase):
    def test_if_else_merge_gets_one_phi(self):
        fn = to_ssa(lower_module(join_if_program())).functions[0]
        merge = fn.blocks[3]
        self.assertEqual(len(merge.phis), 1)
        phi = merge.phis[0]
        self.assertEqual(phi.type, "int")
        labels = [b.label for b in phi.entries]
        self.assertEqual(labels, ["b1", "b2"])
        # The returned value is exactly the phi result.
        self.assertIs(merge.terminator.value, phi.dest)

    def test_phi_lines_name_result_type_and_sources(self):
        text = render_module(to_ssa(lower_module(join_if_program())))
        self.assertIn("b3:", text)
        line = next(
            l for l in text.splitlines() if l.strip().startswith("phi")
            or "= phi" in l
        )
        self.assertTrue(line.strip().startswith("%"))
        self.assertIn(": int = phi ", line)
        self.assertIn("[b1, %", line)
        self.assertIn("[b2, %", line)

    def test_missing_else_phi_carries_fallthrough_value(self):
        # No else branch: the false edge leaves `a` equal to its initial
        # value, and the phi still merges the two incoming definitions.
        module = lower_module(program(func(
            "f", [param("c", "bool"), param("x", "int")], "int",
            [
                let("a", "int", int_(1)),
                if_(var("c"), [assign("a", var("x"))]),
                ret(var("a")),
            ],
        )))
        fn = to_ssa(module).functions[0]
        merge = fn.blocks[3]
        self.assertEqual(len(merge.phis), 1)
        self.assertEqual(
            [b.label for b in merge.phis[0].entries], ["b1", "b2"]
        )

    def test_both_branches_return_yields_no_phi(self):
        module = lower_module(program(func(
            "f", [param("c", "bool")], "int",
            [if_(var("c"), [ret(int_(1))], [ret(int_(2))])],
        )))
        fn = to_ssa(module).functions[0]
        self.assertEqual(len(fn.blocks), 3)
        self.assertFalse(any(b.phis for b in fn.blocks))

    def test_nested_if_phis_at_each_join(self):
        module = lower_module(program(func(
            "h",
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
        fn = to_ssa(module).functions[0]
        phi_blocks = [b.label for b in fn.blocks if b.phis]
        # Inner merge and outer merge each carry a phi.
        self.assertEqual(len(phi_blocks), 2)
        # Each phi's result ultimately feeds the return; the outer phi's
        # inputs include the inner phi result.
        outer = fn.blocks[6].phis[0]
        self.assertIn(fn.blocks[5].phis[0].dest, outer.entries.values())

    def test_loop_header_phis_get_entry_and_backedge_values(self):
        fn = to_ssa(lower_module(program(COUNTER))).functions[0]
        header = fn.blocks[1]
        by_name = {p.name: p.temp for p in fn.params}
        phis = {(phi.entries[fn.blocks[0]], phi.entries[fn.blocks[2]])
                for phi in header.phis}
        # `n`: entry value is the parameter; backedge value is the subtract.
        n_phi = next(
            phi for phi in header.phis
            if phi.entries[fn.blocks[0]] is by_name["n"]
        )
        back_n = n_phi.entries[fn.blocks[2]]
        self.assertIsInstance(back_n, Temp)
        # The backedge value is the subtract result in the body (the last
        # instruction before the jump).
        body = fn.blocks[2]
        sub = next(i for i in reversed(body.instructions)
                   if isinstance(i, BinOp))
        self.assertEqual(sub.operator, "sub")
        self.assertIs(back_n, sub.dest)
        # `acc` similarly merges its initializer with the in-loop add.
        acc_phi = next(phi for phi in header.phis if phi is not n_phi)
        self.assertEqual(
            acc_phi.entries[fn.blocks[2]],
            next(i for i in body.instructions
                 if isinstance(i, BinOp) and i.operator == "add").dest,
        )

    def test_loop_invariant_has_no_header_phi(self):
        # `k` is never reassigned inside the loop: no phi for it.
        module = lower_module(program(func(
            "f", [param("n", "int")], "int",
            [
                let("k", "int", int_(7)),
                while_(compare("gt", var("n"), int_(0)),
                       [assign("n", arith("sub", var("n"), int_(1)))]),
                ret(var("k")),
            ],
        )))
        fn = to_ssa(module).functions[0]
        self.assertFalse(
            any(len(b.phis) > 1 for b in fn.blocks),
            "only n may have a loop phi; k must not",
        )


# --------------------------------------------------------------------------
# Short-circuit handling
# --------------------------------------------------------------------------


class ShortCircuitTests(unittest.TestCase):
    def _logical_function(self, op):
        return program(func(
            "f", [], "bool",
            [
                let("r", "bool",
                    logical(op, bool_(op == "or"),
                            compare("eq", int_(1), int_(1)))),
                ret(var("r")),
            ],
        ))

    def test_and_phi_merges_paths_without_duplicate_write(self):
        fn = to_ssa(lower_module(self._logical_function("and"))).functions[0]
        merge = fn.blocks[3]
        self.assertEqual(len(merge.phis), 1)
        self.assertEqual(
            [b.label for b in merge.phis[0].entries], ["b1", "b2"]
        )
        # The shared non-SSA result temp's two writes collapsed to distinct
        # SSA defs feeding one phi; there is no copy of the result after it.
        self.assertIs(merge.terminator.value, merge.phis[0].dest)

    def test_or_phi_merges_paths(self):
        fn = to_ssa(lower_module(self._logical_function("or"))).functions[0]
        merge = fn.blocks[3]
        self.assertEqual(len(merge.phis), 1)
        self.assertEqual(
            [b.label for b in merge.phis[0].entries], ["b1", "b2"]
        )

    def test_nested_short_circuit_is_well_formed(self):
        # Right operand is itself short-circuit: both merges get phis and
        # every block is terminated (regression for a lowering bug).
        module = lower_module(program(func(
            "f", [], "bool",
            [ret(logical("or", bool_(False),
                         logical("or", bool_(False), bool_(True))))],
        )))
        for f in module.functions:
            self.assertTrue(all(
                b.terminator is not None for b in _reachable(f)
            ))
        fn = to_ssa(module).functions[0]
        assert_ssa_well_formed(self, fn)
        self.assertEqual(len([b for b in fn.blocks if b.phis]), 2)


# --------------------------------------------------------------------------
# Redundant phi pruning
# --------------------------------------------------------------------------


class PruningTests(unittest.TestCase):
    def test_identical_incoming_value_prunes_phi(self):
        # Both branches assign the same SSA value (`x`) to `a`: the copies
        # fold away, both edges carry the same value, no phi remains.
        module = lower_module(program(func(
            "f", [param("c", "bool"), param("x", "int")], "int",
            [
                let("a", "int", var("x")),
                if_(var("c"),
                    [assign("a", var("x"))],
                    [assign("a", var("x"))]),
                ret(var("a")),
            ],
        )))
        fn = to_ssa(module).functions[0]
        self.assertFalse(any(b.phis for b in fn.blocks))

    def test_unused_result_has_no_phi(self):
        # `a` diverges in the branches but is never read afterwards.
        module = lower_module(program(func(
            "g", [param("c", "bool")], "void",
            [
                let("a", "int", int_(1)),
                if_(var("c"),
                    [assign("a", int_(2))],
                    [assign("a", int_(3))]),
            ],
        )))
        fn = to_ssa(module).functions[0]
        self.assertFalse(any(b.phis for b in fn.blocks))

    def test_phi_only_cycle_for_unread_slots_is_removed(self):
        # `a` and `b` only feed each other inside the loop and are never
        # read after it: their header phis reference only one another and
        # the loop backedge, a closed phi-only cycle, so both die.  Only
        # the loop counter `n` keeps a phi.
        module = lower_module(program(func(
            "f", [param("n", "int")], "void",
            [
                let("a", "int", int_(1)),
                let("b", "int", int_(2)),
                while_(compare("gt", var("n"), int_(0)),
                       [
                           assign("a", var("b")),
                           assign("b", var("a")),
                           assign("n", arith("sub", var("n"), int_(1))),
                       ]),
            ],
        )))
        fn = to_ssa(module).functions[0]
        total_phis = sum(len(b.phis) for b in fn.blocks)
        # The single surviving phi is the loop counter's.
        self.assertEqual(total_phis, 1)

    def test_unread_short_circuit_result_has_no_phi(self):
        # The short-circuit result temporary is written on two paths and
        # stored into `r`, but `r` is never read: its merge phi is dead.
        module = lower_module(program(func(
            "f", [], "void",
            [let("r", "bool",
                 logical("and", bool_(True),
                         compare("eq", int_(1), int_(1))))],
        )))
        fn = to_ssa(module).functions[0]
        self.assertFalse(any(b.phis for b in fn.blocks))


# --------------------------------------------------------------------------
# Semantic preservation
# --------------------------------------------------------------------------


class PreservationTests(unittest.TestCase):
    def test_calls_arith_compare_preserved_in_order(self):
        module = lower_module(program(
            func("id", [param("x", "int")], "int", [ret(var("x"))]),
            func("use", [param("a", "int")], "int",
                 [
                     let("b", "int",
                         arith("add", call("id", [var("a")]), int_(1))),
                     # A comparison still appears, used purely as a branch
                     # condition.
                     if_(compare("gt", var("b"), int_(0)),
                         [ret(var("b"))],
                         [ret(int_(0))]),
                 ]),
        ))
        non_ssa = module.functions[1]
        ssa = to_ssa(module).functions[1]

        def signature(fn_):
            calls = [
                (i.name, len(i.args)) for b in fn_.blocks for i in b.instructions
                if isinstance(i, Call)
            ]
            ops = [
                (i.kind, i.operator)
                for b in fn_.blocks for i in b.instructions
                if isinstance(i, BinOp)
            ]
            return calls, ops

        self.assertEqual(signature(non_ssa), signature(ssa))
        # In the entry block the call strictly precedes the add.
        entry_order = [type(i).__name__ for i in ssa.entry.instructions]
        self.assertLess(
            entry_order.index("Call"), entry_order.index("BinOp")
        )

    def test_block_labels_order_and_terminators_preserved(self):
        module = lower_module(program(COUNTER))
        before = module.functions[0]
        after = to_ssa(module).functions[0]
        self.assertEqual(
            [b.id for b in before.blocks], [b.id for b in after.blocks]
        )
        for old, new in zip(before.blocks, after.blocks):
            self.assertEqual(type(old.terminator), type(new.terminator))
            if isinstance(old.terminator, (Jump,)):
                self.assertEqual(
                    old.terminator.target.id, new.terminator.target.id
                )
            elif isinstance(old.terminator, Branch):
                self.assertEqual(
                    old.terminator.true_target.id,
                    new.terminator.true_target.id,
                )
                self.assertEqual(
                    old.terminator.false_target.id,
                    new.terminator.false_target.id,
                )

    def test_function_order_preserved(self):
        module = lower_module(program(
            func("a", [], "void", []),
            func("b", [], "int", [ret(int_(1))]),
            func("c", [], "void", []),
        ))
        self.assertEqual(
            [f.name for f in to_ssa(module).functions], ["a", "b", "c"]
        )


# --------------------------------------------------------------------------
# Determinism and idempotence
# --------------------------------------------------------------------------


COUNTER = func(
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
)


class DeterminismTests(unittest.TestCase):
    PROGRAMS = [
        program(COUNTER),
        join_if_program(),
        program(func(
            "f", [], "bool",
            [let("r", "bool",
                 logical("and", bool_(True),
                         compare("eq", int_(1), int_(1)))),
             ret(var("r"))])),
    ]

    def test_repeated_conversion_byte_identical(self):
        for ast in self.PROGRAMS:
            module = lower_module(ast)
            first = render_module(to_ssa(module))
            second = render_module(to_ssa(module))
            self.assertEqual(first, second)

    def test_idempotent_structurally_and_textually(self):
        for ast in self.PROGRAMS:
            module = lower_module(ast)
            once = to_ssa(module)
            twice = to_ssa(once)
            thrice = to_ssa(twice)
            self.assertEqual(
                render_module(once), render_module(twice)
            )
            self.assertEqual(
                render_module(twice), render_module(thrice)
            )
            for f1, f2 in zip(once.functions, twice.functions):
                self.assertEqual(
                    [len(b.phis) for b in f1.blocks],
                    [len(b.phis) for b in f2.blocks],
                )
                self.assertEqual(
                    [d.id for d in _defined_temps(f1)],
                    [d.id for d in _defined_temps(f2)],
                )
            # The already-SSA module is copied, not returned by identity.
            self.assertIsNot(twice, once)

    def test_value_numbers_start_at_zero_per_function(self):
        module = lower_module(program(
            func("a", [param("x", "int")], "int",
                 [ret(arith("add", var("x"), int_(1)))]),
            func("b", [param("y", "int")], "int",
                 [ret(arith("sub", var("y"), int_(2)))]),
        ))
        for fn_ in to_ssa(module).functions:
            self.assertEqual(
                min(d.id for d in _defined_temps(fn_)), 0
            )
            assert_ssa_well_formed(self, fn_)

    def test_ssa_rendering_is_stable_text(self):
        text = render_module(to_ssa(lower_module(program(COUNTER))))
        self.assertIn("function counter(n: int @ %0) -> int {", text)
        # Phi line carries the type and both labelled sources.
        self.assertIn(": int = phi [b0, %0], [b2, %", text)
        # No locals section and no %v slots in SSA output.
        self.assertNotIn("locals:", text)
        self.assertNotIn("%v", text)
        self.assertNotIn("%t", text)


# --------------------------------------------------------------------------
# Backward compatibility of the existing pipeline
# --------------------------------------------------------------------------


class CompatibilityTests(unittest.TestCase):
    def test_non_ssa_rendering_unchanged(self):
        text = emit_ir(program(COUNTER))
        self.assertIn("%t0: int = const 0", text)
        self.assertIn("%v1: int = copy %t0", text)
        self.assertIn("locals:", text)

    def test_non_ssa_module_has_no_phi_field_populated(self):
        module = lower_module(program(COUNTER))
        self.assertFalse(module.ssa)
        self.assertTrue(all(
            block.phis == [] for block in module.functions[0].blocks
        ))

    def test_while_short_circuit_condition_lowers_well_formed(self):
        # Regression: a short-circuit while condition used to strand
        # unterminated, orphan blocks.
        module = lower_module(program(func(
            "f", [param("x", "int")], "void",
            [while_(
                logical("and", compare("gt", var("x"), int_(0)),
                        compare("lt", var("x"), int_(10))),
                [assign("x", arith("sub", var("x"), int_(1)))])],
        )))
        for fn_ in module.functions:
            self.assertTrue(all(
                b.terminator is not None for b in _reachable(fn_)
            ))
        # And the condition value merges into a phi-free header branch.
        ssa = to_ssa(module).functions[0]
        assert_ssa_well_formed(self, ssa)


if __name__ == "__main__":
    unittest.main()
