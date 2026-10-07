"""Tests for the library-level pipeline entry :func:`optimize_module`.

The single-entry driver is pinned on five contracts:

1. Validation -- everything is checked before any pass runs.  Non-Module
   inputs and malformed ``passes`` raise ``TypeError``; unknown names
   (including ``render_module``) and fold/dce scheduled before the SSA
   stage on a non-SSA input raise ``ValueError``; rejected calls never
   touch the input.
2. Independence -- the result is a brand new module sharing no mutable
   function, block, instruction or phi container with the input, even for
   the empty schedule and even for the empty module; an empty schedule
   preserves the input's SSA/non-SSA flavor.
3. Orchestration -- omitted ``passes`` is exactly
   ``ssa, fold, licm, dce, ssa`` (the hand-written chain renders
   byte-identically) and explicit schedules execute their names in order
   with repeats.
4. Determinism / idempotence -- repeated calls and re-applying the default
   schedule give structurally and textually identical modules.
5. Semantics -- across legal schedules (including SSA-input schedules that
   start with fold/dce, and the empty schedule), the executed return value,
   ordered call trace/arguments and division/modulo fault sites agree with
   the unoptimized baseline.
"""
import copy
import unittest

from compiler_ir import (
    Module,
    eliminate_dead_code,
    emit_ir,
    fold_constants,
    hoist_loop_invariants,
    lower_module,
    optimize_module,
    render_module,
    to_ssa,
)

from test_pass_ordering import (
    DEFAULT_ORDER,
    _CASES,
    _clean_program,
    _folded_zero_fault_program,
    _interpret,
    _loop_program,
    structural_signature,
)
from test_pipeline import func, param, program, ret, var
from test_dce import _container_objects


# Explicit schedules exercised from a *non-SSA* input.
NON_SSA_ORDERS = [
    DEFAULT_ORDER,
    ("ssa", "fold", "fold", "licm", "dce", "ssa"),
    ("ssa", "fold", "licm", "dce", "dce", "ssa"),
    ("ssa", "dce", "fold", "licm", "dce", "ssa"),
    ("ssa", "fold", "dce", "fold", "licm", "dce", "ssa"),
    ("ssa",),
    ("ssa", "fold"),
    ("ssa", "dce"),
    ("ssa", "fold", "dce"),
    ("ssa", "dce", "fold"),
    ("ssa", "fold", "licm"),
    ("ssa", "licm"),
    ("ssa", "ssa", "fold", "licm", "dce", "ssa"),
    ("ssa", "fold", "ssa", "licm", "dce", "ssa"),
    ("ssa", "fold", "licm", "licm", "dce", "ssa"),
]

# Schedules exercised from an already-SSA input: fold/licm/dce may lead,
# ssa may re-canonicalize, names may repeat.
SSA_ORDERS = [
    None,
    ("fold", "licm", "dce", "ssa"),
    ("fold",),
    ("dce",),
    ("licm",),
    ("ssa",),
    ("dce", "fold", "licm", "dce", "ssa"),
    ("fold", "fold", "dce"),
    ("licm", "licm"),
    ("dce", "dce"),
    ("ssa", "fold", "licm", "dce", "ssa"),
    (),
]


def _lowered(builder):
    return lower_module(copy.deepcopy(builder()))


class OptimizeValidationTests(unittest.TestCase):
    def setUp(self):
        self.lowered = _lowered(_clean_program)
        self.ssa = to_ssa(self.lowered)
        self.lowered_text = render_module(self.lowered)
        self.ssa_text = render_module(self.ssa)

    def assert_unchanged(self):
        self.assertEqual(render_module(self.lowered), self.lowered_text)
        self.assertEqual(render_module(self.ssa), self.ssa_text)

    # -- TypeError ----------------------------------------------------------

    def test_module_must_be_a_module(self):
        for bad in (None, 42, "module", {}, [], object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    optimize_module(bad)
                with self.assertRaises(TypeError):
                    optimize_module(bad, ())
                with self.assertRaises(TypeError):
                    optimize_module(bad, ("fold",))

    def test_passes_string_is_type_error_not_letter_sequence(self):
        with self.assertRaises(TypeError):
            optimize_module(self.lowered, "fold")
        with self.assertRaises(TypeError):
            optimize_module(self.ssa, "ssa")
        with self.assertRaises(TypeError):
            optimize_module(self.lowered, "")  # even an empty string

    def test_passes_non_sequence_is_type_error(self):
        for bad in (42, 1.5, object(), (x for x in ("ssa",))):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    optimize_module(self.lowered, bad)

    def test_passes_with_non_string_elements_is_type_error(self):
        for bad in (["fold", 1], ("ssa", None), [("ssa",)], [b"ssa"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    optimize_module(self.lowered, bad)
                with self.assertRaises(TypeError):
                    optimize_module(self.ssa, bad)
        self.assert_unchanged()

    # -- ValueError ---------------------------------------------------------

    def test_unknown_pass_name_is_value_error(self):
        for name in ("inline", "render_module", "render", "isel",
                     "eliminate_dead_code", "to_ssa", "FOLD", ""):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    optimize_module(self.lowered, ("ssa", name))
                with self.assertRaises(ValueError):
                    optimize_module(self.ssa, (name,))
        self.assert_unchanged()

    def test_render_module_is_unknown_even_last(self):
        with self.assertRaises(ValueError):
            optimize_module(self.lowered, ("ssa", "fold", "dce",
                                           "render_module"))

    def test_fold_dce_before_ssa_rejected_on_non_ssa_input(self):
        for order in (("fold",), ("dce",), ("licm",),
                      ("fold", "ssa"), ("dce", "fold", "ssa"),
                      ("licm", "fold", "ssa"),
                      ("fold", "dce"), ("dce", "dce"),
                      ("fold", "licm", "dce", "ssa")):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    optimize_module(self.lowered, order)

    def test_ssa_input_may_start_with_fold_licm_or_dce(self):
        for order in (("fold",), ("dce",), ("licm",),
                      ("fold", "licm", "dce", "ssa"),
                      ("dce", "fold"), ()):
            with self.subTest(order=order):
                optimize_module(self.ssa, order)  # must not raise
        self.assert_unchanged()

    def test_late_unknown_name_still_prevalidated(self):
        # A bad final entry must be caught before any earlier valid pass
        # runs as a side effect.
        with self.assertRaises(ValueError):
            optimize_module(self.lowered, ("ssa", "fold", "dce", "bogus"))
        self.assert_unchanged()

    # -- input integrity on failure ----------------------------------------

    def test_failed_validation_leaves_input_byte_identical(self):
        failing = [
            ("non-module", lambda m: optimize_module(42, ("ssa",))),
            ("string passes", lambda m: optimize_module(m, "ssa")),
            ("non-string element",
             lambda m: optimize_module(m, ("ssa", 3))),
            ("unknown name",
             lambda m: optimize_module(m, ("ssa", "bogus"))),
            ("unknown render",
             lambda m: optimize_module(m, ("render_module",))),
            ("precondition",
             lambda m: optimize_module(m, ("fold", "dce"))),
            ("late unknown name",
             lambda m: optimize_module(m, ("ssa", "fold", "dce",
                                           "bogus"))),
        ]
        original_blocks = [
            id(block)
            for fn_ in self.lowered.functions for block in fn_.blocks
        ]
        for label, call in failing:
            with self.subTest(label=label):
                with self.assertRaises((TypeError, ValueError)):
                    call(self.lowered)
        self.assertEqual(render_module(self.lowered), self.lowered_text)
        self.assertEqual(
            [id(block) for fn_ in self.lowered.functions
             for block in fn_.blocks],
            original_blocks)


class OptimizeIndependenceTests(unittest.TestCase):
    def test_default_returns_new_ssa_module_with_disjoint_containers(self):
        for builder in (_clean_program, _loop_program):
            lowered = _lowered(builder)
            result = optimize_module(lowered)
            with self.subTest(builder=builder.__name__):
                self.assertIsNot(result, lowered)
                self.assertTrue(result.ssa)
                self.assertFalse(lowered.ssa)
                self.assertTrue(
                    _container_objects(lowered).isdisjoint(
                        _container_objects(result)))

    def test_ssa_input_also_gets_disjoint_containers(self):
        lowered = _lowered(_clean_program)
        ssa = to_ssa(lowered)
        for order in SSA_ORDERS:
            result = optimize_module(ssa, order)
            with self.subTest(order=order):
                self.assertIsNot(result, ssa)
                self.assertTrue(result.ssa)
                self.assertTrue(
                    _container_objects(ssa).isdisjoint(
                        _container_objects(result)))

    def test_empty_schedule_non_ssa_is_independent_equivalent_copy(self):
        lowered = _lowered(_loop_program)
        lowered_text = render_module(lowered)
        copied = optimize_module(lowered, ())
        self.assertFalse(copied.ssa)
        self.assertEqual(render_module(copied), lowered_text)
        self.assertTrue(
            _container_objects(lowered).isdisjoint(_container_objects(copied)))

        # Deep independence: mutating the copy's containers cannot reach
        # the input module.
        copy_block = copied.functions[0].blocks[0]
        copy_block.instructions.append(copy_block.instructions[0])
        copied.functions.append(copied.functions[0])
        self.assertNotEqual(len(copied.functions), len(lowered.functions))
        self.assertEqual(render_module(lowered), lowered_text)

    def test_empty_schedule_ssa_is_independent_equivalent_copy(self):
        ssa = to_ssa(_lowered(_clean_program))
        copied = optimize_module(ssa, [])
        self.assertTrue(copied.ssa)
        self.assertEqual(render_module(copied), render_module(ssa))
        self.assertTrue(
            _container_objects(ssa).isdisjoint(_container_objects(copied)))

    def test_empty_module_every_schedule(self):
        for ssa_flag, order in ((False, ()), (True, ()), (True, None),
                               (True, ("fold",)), (True, ("dce",)),
                               (True, ("licm",)), (True, ("ssa",))):
            empty = Module([], ssa=ssa_flag)
            result = optimize_module(empty, order)
            with self.subTest(ssa=ssa_flag, order=order):
                self.assertIsNot(result, empty)
                self.assertEqual(result.functions, [])
                self.assertEqual(result.ssa,
                                 ssa_flag if order == () else True)

    def test_input_never_mutated_for_any_schedule(self):
        for builder in (_clean_program, _loop_program,
                        _folded_zero_fault_program):
            if builder is _folded_zero_fault_program:
                ast = builder("div")
                lowered = lower_module(copy.deepcopy(ast))
            else:
                lowered = _lowered(builder)
            text_before = render_module(lowered)
            for order in NON_SSA_ORDERS:
                optimize_module(lowered, list(order))
                with self.subTest(builder=builder.__name__, order=order):
                    self.assertEqual(render_module(lowered), text_before)

    def test_list_and_tuple_passes_equivalent(self):
        lowered = _lowered(_clean_program)
        as_tuple = optimize_module(lowered, DEFAULT_ORDER)
        as_list = optimize_module(lowered, list(DEFAULT_ORDER))
        self.assertEqual(render_module(as_tuple), render_module(as_list))


class OptimizeOrchestrationTests(unittest.TestCase):
    def test_default_order_matches_manual_chain(self):
        for builder in (_clean_program, _loop_program):
            lowered = _lowered(builder)
            driven = optimize_module(lowered)
            manual = to_ssa(
                eliminate_dead_code(
                    hoist_loop_invariants(
                        fold_constants(to_ssa(lowered)))))
            with self.subTest(builder=builder.__name__):
                self.assertEqual(render_module(driven), render_module(manual))
                self.assertEqual(
                    structural_signature(driven),
                    structural_signature(manual))

    def test_explicit_default_equals_omitted(self):
        lowered = _lowered(_loop_program)
        self.assertEqual(
            render_module(optimize_module(lowered)),
            render_module(optimize_module(lowered, DEFAULT_ORDER)),
        )

    def test_repeated_names_run_in_order(self):
        lowered = _lowered(_clean_program)
        # fold is a fixed point: repeating it must not change the endpoint
        # versus the same schedule without the repeat.
        once = optimize_module(lowered, ("ssa", "fold", "dce", "ssa"))
        twice = optimize_module(
            lowered, ("ssa", "fold", "fold", "dce", "ssa"))
        self.assertEqual(render_module(once), render_module(twice))

    def test_result_is_traversable_and_renderable_but_text_not_rendered(self):
        lowered = _lowered(_clean_program)
        result = optimize_module(lowered)
        # Plain module object, no text side channel stored on it.
        self.assertIsInstance(result, Module)
        self.assertTrue(all(
            block.terminator is not None
            for fn_ in result.functions for block in fn_.blocks))
        self.assertIsInstance(render_module(result), str)


class DeterminismIdempotenceTests(unittest.TestCase):
    def test_same_module_same_order_byte_identical(self):
        for builder in (_clean_program, _loop_program):
            lowered = _lowered(builder)
            for order in NON_SSA_ORDERS:
                runs = [
                    render_module(optimize_module(lowered, order)).encode()
                    for _ in range(3)
                ]
                with self.subTest(builder=builder.__name__, order=order):
                    self.assertEqual(runs[0], runs[1])
                    self.assertEqual(runs[1], runs[2])

    def test_default_reapplied_to_its_own_result_is_stable(self):
        for builder in (_clean_program, _loop_program):
            lowered = _lowered(builder)
            once = optimize_module(lowered)
            twice = optimize_module(once)
            thrice = optimize_module(twice)
            with self.subTest(builder=builder.__name__):
                self.assertEqual(
                    render_module(once), render_module(twice))
                self.assertEqual(
                    render_module(twice), render_module(thrice))
                self.assertEqual(
                    structural_signature(once),
                    structural_signature(twice))
                self.assertTrue(
                    _container_objects(once).isdisjoint(
                        _container_objects(twice)))

    def test_empty_schedule_is_stable_and_disjoint(self):
        lowered = _lowered(_loop_program)
        first = optimize_module(lowered, ())
        second = optimize_module(first, ())
        self.assertEqual(render_module(first), render_module(second))
        self.assertTrue(
            _container_objects(first).isdisjoint(_container_objects(second)))


class OptimizeSemanticsTests(unittest.TestCase):
    def _check_orders(self, ast, entry, arguments, baseline, orders,
                      ssa_input=False):
        lowered = lower_module(copy.deepcopy(ast))
        if ssa_input:
            seed = to_ssa(lowered)
            for order in orders:
                result = optimize_module(seed, order)
                with self.subTest(entry=entry, args=arguments, order=order,
                                  flavor="ssa-input"):
                    outcome = _interpret(result, entry, arguments)
                    self.assertEqual(baseline, outcome)
        else:
            for order in orders:
                result = optimize_module(lowered, order)
                with self.subTest(entry=entry, args=arguments, order=order,
                                  flavor="non-ssa-input"):
                    outcome = _interpret(result, entry, arguments)
                    self.assertEqual(
                        baseline, outcome,
                        msg=(f"order {order!r} changed observables: "
                             f"baseline={baseline!r} actual={outcome!r}"))

    def test_every_schedule_preserves_observables(self):
        for label, builder, entry, arguments, pinned in _CASES:
            lowered = lower_module(copy.deepcopy(builder()))
            baseline = _interpret(lowered, entry, arguments)
            self.assertEqual(baseline, pinned)
            with self.subTest(sample=label):
                self._check_orders(
                    builder(), entry, arguments, baseline, NON_SSA_ORDERS)
                self._check_orders(
                    builder(), entry, arguments, baseline, SSA_ORDERS,
                    ssa_input=True)

    def test_empty_non_ssa_schedule_runs_as_non_ssa(self):
        for label, builder, entry, arguments, pinned in _CASES:
            lowered = lower_module(copy.deepcopy(builder()))
            copied = optimize_module(lowered, ())
            with self.subTest(sample=label):
                self.assertFalse(copied.ssa)
                self.assertEqual(
                    _interpret(copied, entry, arguments), pinned)

    def test_side_effecting_calls_survive_every_order(self):
        lowered = _lowered(_clean_program)

        def calls_of(result):
            caller = next(
                fn_ for fn_ in result.functions if fn_.name == "clean")
            return [
                ins.name for block in caller.blocks
                for ins in block.instructions
                if type(ins).__name__ == "Call"
            ]

        for order in NON_SSA_ORDERS:
            with self.subTest(order=order, flavor="non-ssa-input"):
                self.assertEqual(
                    calls_of(optimize_module(lowered, order)),
                    ["observe", "observe"])

        ssa_seed = to_ssa(lowered)
        for order in SSA_ORDERS:
            with self.subTest(order=order, flavor="ssa-input"):
                self.assertEqual(
                    calls_of(optimize_module(ssa_seed, order)),
                    ["observe", "observe"])


class ExistingSurfaceTests(unittest.TestCase):
    def test_optimize_module_is_public(self):
        import compiler_ir

        self.assertIn("optimize_module", compiler_ir.__all__)
        self.assertIs(compiler_ir.optimize_module, optimize_module)

    def test_hoist_loop_invariants_is_public(self):
        import compiler_ir

        self.assertIn("hoist_loop_invariants", compiler_ir.__all__)
        self.assertIs(
            compiler_ir.hoist_loop_invariants, hoist_loop_invariants)
        self.assertIn("licm", optimize_module.__doc__ + "")
        from compiler_ir.pipeline import DEFAULT_PASSES, PASS_NAMES

        self.assertEqual(
            DEFAULT_PASSES, ("ssa", "fold", "licm", "dce", "ssa"))
        self.assertIn("licm", PASS_NAMES)

    def test_emit_ir_remains_unoptimized_lowering(self):
        from test_pipeline import arith, int_, let

        ast = program(
            func("f", [param("x", "int")], "int",
                 [let("y", "int", arith("add", var("x"), int_(1))),
                  ret(var("y"))]),
        )
        text = emit_ir(ast)
        self.assertTrue(text.startswith("module\n"))
        self.assertIn("locals:", text)
        self.assertNotIn("phi", text)
        self.assertEqual(text, emit_ir(ast))

    def test_cli_still_only_version_and_help(self):
        from compiler_ir.__main__ import main

        self.assertEqual(main(["version"]), 0)
        self.assertEqual(main(["help"]), 0)
        self.assertEqual(main(["optimize"]), 2)


if __name__ == "__main__":
    unittest.main()
