"""Tests for the library-level pipeline entry point :func:`optimize_module`.

The individual passes (``to_ssa``, ``fold_constants``,
``eliminate_dead_code``) and the manual scheduling harness are pinned by
their own suites; this module pins the orchestration itself:

* the API contract -- accepted inputs, the default sequence, explicit
  sequences with repeats, and the empty-sequence copy;
* up-front validation -- ``TypeError`` for a non-module or a malformed
  ``passes`` argument, ``ValueError`` for unknown names (including
  ``"render_module"``) and for ``fold``/``dce`` scheduled before the
  first ``ssa`` on non-SSA input, all checked before anything runs;
* isolation -- the input is never mutated and no mutable function, block,
  instruction or phi container is shared with the result, not even for
  the empty module or the empty sequence;
* determinism and idempotence -- same input and order give byte-identical
  renderings, and the default sequence is a fixed point of itself;
* semantics -- every legal order preserves the observable behavior
  (return value, ordered call trace, runtime faults) of the unoptimized
  baseline, reusing the interpreter from ``test_pass_ordering``.
"""
import copy
import unittest

from compiler_ir import (
    DEFAULT_PASSES,
    Module,
    eliminate_dead_code,
    fold_constants,
    lower_module,
    optimize_module,
    render_module,
    to_ssa,
)

from test_dce import _container_objects
from test_pass_ordering import (
    _CASES,
    _interpret,
    _join_program,
    _clean_program,
    _loop_program,
    structural_signature,
)
from test_pipeline import program


# Legal explicit orders beyond the default, exercised end to end.
EXPLICIT_ORDERS = (
    ("ssa",),
    ("ssa", "fold"),
    ("ssa", "dce"),
    ("ssa", "fold", "dce"),
    ("ssa", "dce", "fold"),
    ("ssa", "fold", "fold", "dce", "ssa"),
    ("ssa", "fold", "dce", "dce", "ssa"),
    ("ssa", "dce", "fold", "dce", "ssa"),
    ("ssa", "fold", "dce", "fold", "dce", "ssa"),
    ("ssa", "ssa"),
)

# Orders legal on an already-SSA input (may start with fold/dce).
SSA_INPUT_ORDERS = (
    ("fold",),
    ("dce",),
    ("fold", "dce", "ssa"),
    ("dce", "fold", "dce", "ssa"),
    ("ssa", "fold", "dce", "ssa"),
)


def _lowered(builder):
    return lower_module(copy.deepcopy(builder()))


class ApiContractTests(unittest.TestCase):
    def test_default_matches_manual_pipeline(self):
        for label, builder, _entry, _args, _pin in _CASES:
            with self.subTest(sample=label):
                lowered = _lowered(builder)
                manual = to_ssa(
                    eliminate_dead_code(fold_constants(to_ssa(lowered)))
                )
                optimized = optimize_module(lowered)
                self.assertTrue(optimized.ssa)
                self.assertEqual(
                    render_module(manual), render_module(optimized))
                self.assertEqual(
                    structural_signature(manual),
                    structural_signature(optimized),
                )

    def test_default_passes_constant_is_the_documented_sequence(self):
        self.assertEqual(DEFAULT_PASSES, ("ssa", "fold", "dce", "ssa"))

    def test_none_and_omitted_passes_are_the_default(self):
        lowered = _lowered(_loop_program)
        self.assertEqual(
            render_module(optimize_module(lowered)),
            render_module(optimize_module(lowered, None)),
        )

    def test_explicit_orders_run_in_the_given_sequence(self):
        manual_funcs = {
            "ssa": to_ssa,
            "fold": fold_constants,
            "dce": eliminate_dead_code,
        }
        for order in EXPLICIT_ORDERS:
            with self.subTest(order=order):
                lowered = _lowered(_join_program)
                expected = lowered
                for name in order:
                    expected = manual_funcs[name](expected)
                actual = optimize_module(lowered, order)
                self.assertEqual(
                    render_module(expected), render_module(actual))
                self.assertEqual(
                    structural_signature(expected),
                    structural_signature(actual),
                )

    def test_explicit_order_accepts_list_and_tuple(self):
        lowered = _lowered(_clean_program)
        as_tuple = optimize_module(lowered, ("ssa", "fold", "dce", "ssa"))
        as_list = optimize_module(lowered, ["ssa", "fold", "dce", "ssa"])
        self.assertEqual(render_module(as_tuple), render_module(as_list))

    def test_ssa_input_may_start_with_fold_or_dce(self):
        ssa = to_ssa(_lowered(_clean_program))
        for order in SSA_INPUT_ORDERS:
            with self.subTest(order=order):
                result = optimize_module(ssa, order)
                self.assertTrue(result.ssa)
                # The same sequence reaches the same module whether the
                # harness or the caller built the SSA input.
                from_non_ssa = optimize_module(
                    _lowered(_clean_program), ("ssa",) + order)
                self.assertEqual(
                    render_module(from_non_ssa), render_module(result))

    def test_result_is_traversable_and_renderable(self):
        optimized = optimize_module(_lowered(_loop_program))
        text = render_module(optimized)
        self.assertTrue(text.endswith("\n"))
        self.assertIn("function loopsum(", text)
        self.assertTrue(optimized.functions)
        self.assertIs(
            optimized.functions[0].entry, optimized.functions[0].blocks[0])


class ValidationTests(unittest.TestCase):
    def test_non_module_raises_type_error(self):
        for bad in (None, 1, "module", [], {}, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    optimize_module(bad)
        # The module check wins even over a malformed passes argument.
        with self.assertRaises(TypeError):
            optimize_module("module", "ssa")

    def test_malformed_passes_raise_type_error(self):
        lowered = _lowered(_clean_program)
        for bad in ("ssa", b"ssa", 42, 3.5, object(), {"ssa", "fold"},
                    {"ssa": 1}):
            with self.subTest(passes=bad):
                with self.assertRaises(TypeError):
                    optimize_module(lowered, bad)
        # Non-string elements, including None and nested sequences.
        for bad in ((1,), ("ssa", None), ("ssa", b"fold"), (["ssa"],),
                    ("ssa", 2)):
            with self.subTest(passes=bad):
                with self.assertRaises(TypeError):
                    optimize_module(lowered, bad)

    def test_unknown_pass_name_raises_value_error(self):
        lowered = _lowered(_clean_program)
        for bad in (("bogus",), ("ssa", "inline"), ("SSA",), ("",),
                    ("ssa", "fold ",), ("isel",), ("render",),
                    ("render_module",), ("ssa", "render_module")):
            with self.subTest(passes=bad):
                with self.assertRaises(ValueError):
                    optimize_module(lowered, bad)

    def test_render_module_is_not_a_schedulable_pass(self):
        lowered = _lowered(_clean_program)
        with self.assertRaises(ValueError):
            optimize_module(lowered, ("ssa", "fold", "dce", "render_module"))
        ssa = to_ssa(lowered)
        with self.assertRaises(ValueError):
            optimize_module(ssa, ("render_module",))

    def test_fold_and_dce_before_ssa_raise_value_error(self):
        lowered = _lowered(_clean_program)
        self.assertFalse(lowered.ssa)
        for bad in (("fold",), ("dce",), ("fold", "ssa"), ("dce", "ssa"),
                    ("fold", "dce", "ssa"), ("dce", "fold", "ssa")):
            with self.subTest(passes=bad):
                with self.assertRaises(ValueError):
                    optimize_module(lowered, bad)

    def test_validation_happens_before_any_pass_runs(self):
        # A bad name late in the sequence must fail the whole call even
        # though the leading passes are legal.
        lowered = _lowered(_clean_program)
        before = render_module(lowered)
        with self.assertRaises(ValueError):
            optimize_module(lowered, ("ssa", "fold", "bogus"))
        with self.assertRaises(ValueError):
            optimize_module(lowered, ("ssa", "fold", "dce", "render_module"))
        self.assertEqual(render_module(lowered), before)

    def test_failed_validation_leaves_input_untouched(self):
        lowered = _lowered(_loop_program)
        before = render_module(lowered)
        for call in (
            lambda: optimize_module(lowered, "ssa"),
            lambda: optimize_module(lowered, ("ssa", 1)),
            lambda: optimize_module(lowered, ("fold",)),
            lambda: optimize_module(lowered, ("ssa", "bogus")),
        ):
            with self.assertRaises((TypeError, ValueError)):
                call()
            self.assertEqual(render_module(lowered), before)


class IsolationTests(unittest.TestCase):
    def test_input_is_never_mutated(self):
        for label, builder, _entry, _args, _pin in _CASES:
            with self.subTest(sample=label):
                lowered = _lowered(builder)
                before = render_module(lowered)
                optimize_module(lowered)
                optimize_module(lowered, ("ssa", "fold"))
                optimize_module(lowered, ())
                self.assertEqual(render_module(lowered), before)

    def test_result_shares_no_containers_with_input(self):
        for label, builder, _entry, _args, _pin in _CASES:
            with self.subTest(sample=label):
                lowered = _lowered(builder)
                for order in (None, ("ssa",), ("ssa", "fold", "dce"), ()):
                    result = optimize_module(lowered, order)
                    self.assertIsNot(result, lowered)
                    self.assertTrue(
                        _container_objects(lowered).isdisjoint(
                            _container_objects(result)),
                        msg=f"{label!r} order {order!r}: shared containers",
                    )

    def test_empty_sequence_returns_independent_equivalent_copy(self):
        for label, builder, _entry, _args, _pin in _CASES:
            with self.subTest(sample=label):
                lowered = _lowered(builder)
                copied = optimize_module(lowered, ())
                self.assertIsNot(copied, lowered)
                self.assertEqual(copied.ssa, lowered.ssa)
                self.assertEqual(
                    render_module(lowered), render_module(copied))
                self.assertTrue(
                    _container_objects(lowered).isdisjoint(
                        _container_objects(copied)))

                ssa = to_ssa(lowered)
                ssa_copied = optimize_module(ssa, [])
                self.assertIsNot(ssa_copied, ssa)
                self.assertTrue(ssa_copied.ssa)
                self.assertEqual(render_module(ssa), render_module(ssa_copied))
                self.assertTrue(
                    _container_objects(ssa).isdisjoint(
                        _container_objects(ssa_copied)))

    def test_empty_module_and_empty_sequence_stay_independent(self):
        for empty in (Module([], ssa=False), Module([], ssa=True)):
            with self.subTest(ssa=empty.ssa):
                copied = optimize_module(empty, ())
                self.assertIsNot(copied, empty)
                self.assertEqual(copied.ssa, empty.ssa)
                self.assertEqual(copied.functions, [])
                self.assertEqual(
                    render_module(empty), render_module(copied))
                optimized = optimize_module(empty)
                self.assertIsNot(optimized, empty)
                self.assertTrue(optimized.ssa)
                self.assertEqual(optimized.functions, [])

    def test_successive_calls_do_not_alias_each_other(self):
        lowered = _lowered(_clean_program)
        first = optimize_module(lowered)
        second = optimize_module(lowered)
        self.assertIsNot(first, second)
        self.assertTrue(
            _container_objects(first).isdisjoint(
                _container_objects(second)))


class DeterminismAndIdempotenceTests(unittest.TestCase):
    def test_repeated_calls_are_byte_identical(self):
        for label, builder, _entry, _args, _pin in _CASES:
            ast = builder()
            for order in (None, ("ssa", "fold", "dce"),
                          ("ssa", "dce", "fold", "dce", "ssa"), ()):
                with self.subTest(sample=label, order=order):
                    texts = [
                        render_module(optimize_module(
                            lower_module(copy.deepcopy(ast)), order))
                        for _ in range(3)
                    ]
                    self.assertEqual(texts[0], texts[1])
                    self.assertEqual(texts[1], texts[2])
                    encoded = [t.encode("utf-8") for t in texts]
                    self.assertEqual(encoded[0], encoded[1])
                    self.assertEqual(encoded[1], encoded[2])

    def test_default_sequence_is_a_fixed_point_of_itself(self):
        for label, builder, _entry, _args, _pin in _CASES:
            with self.subTest(sample=label):
                once = optimize_module(_lowered(builder))
                twice = optimize_module(once)
                thrice = optimize_module(twice)
                self.assertEqual(
                    render_module(once), render_module(twice))
                self.assertEqual(
                    render_module(twice), render_module(thrice))
                self.assertEqual(
                    structural_signature(once),
                    structural_signature(twice),
                )


class SemanticsTests(unittest.TestCase):
    def test_every_legal_order_matches_the_unoptimized_baseline(self):
        for label, builder, entry, arguments, pinned in _CASES:
            ast = builder()
            with self.subTest(sample=label):
                baseline = _interpret(
                    lower_module(copy.deepcopy(ast)), entry, arguments)
                self.assertEqual(pinned, baseline)
                # Orders legal on the non-SSA lowered module.
                for order in EXPLICIT_ORDERS + (None,):
                    with self.subTest(order=order):
                        optimized = optimize_module(
                            lower_module(copy.deepcopy(ast)), order)
                        self.assertEqual(baseline,
                                         _interpret(optimized, entry,
                                                    arguments))
                # Orders that lead with fold/dce are legal on SSA input.
                for order in SSA_INPUT_ORDERS:
                    with self.subTest(order=order, input="ssa"):
                        ssa = to_ssa(lower_module(copy.deepcopy(ast)))
                        optimized = optimize_module(ssa, order)
                        self.assertEqual(baseline,
                                         _interpret(optimized, entry,
                                                    arguments))

    def test_side_effecting_calls_survive_every_order(self):
        # The default pipeline must keep both calls of the clean sample in
        # their original relative order.
        optimized = optimize_module(_lowered(_clean_program))
        caller = next(f for f in optimized.functions if f.name == "clean")
        from compiler_ir import Call
        calls = [ins.name for b in caller.blocks for ins in b.instructions
                 if isinstance(ins, Call)]
        self.assertEqual(calls, ["observe", "observe"])


class ExistingBehaviorTests(unittest.TestCase):
    def test_emit_ir_is_unchanged(self):
        from compiler_ir import emit_ir
        ast = _join_program()
        self.assertEqual(emit_ir(ast), render_module(lower_module(ast)))
        self.assertIn("locals:", emit_ir(_join_program()))

    def test_cli_still_only_offers_version_and_help(self):
        import subprocess
        import sys
        for args, code in ((["version"], 0), (["help"], 0), ([], 0),
                           (["optimize"], 2)):
            with self.subTest(args=args):
                completed = subprocess.run(
                    [sys.executable, "-m", "compiler_ir", *args],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(completed.returncode, code)


if __name__ == "__main__":
    unittest.main()
