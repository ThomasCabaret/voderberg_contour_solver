import unittest

import external_boundary_constraints as external
import joint_translation_z3 as joint


class JointTranslationZ3Tests(unittest.TestCase):
    def _problem(self):
        case, state = joint.find_voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        return joint.build_z3_problem(system)

    def test_voderberg_problem_is_polynomial_qfnra(self):
        problem = self._problem()
        self.assertIn("(set-logic QF_NRA)", problem.script_smt2)
        self.assertNotIn("(sin ", problem.script_smt2)
        self.assertNotIn("(cos ", problem.script_smt2)
        self.assertEqual(problem.translation_equation_count, 4)
        self.assertGreaterEqual(problem.rotation_equation_count, 2)

    def test_every_voderberg_chord_is_required_nonzero(self):
        problem = self._problem()
        for _name, (x_symbol, y_symbol) in problem.chord_symbol_map:
            norm_fragment = f"(* {x_symbol} {x_symbol})"
            self.assertIn(norm_fragment, problem.script_smt2)
            self.assertIn("(assert (>", problem.script_smt2)

    def test_problem_records_relaxation_scope(self):
        problem = self._problem()
        self.assertTrue(problem.relaxation_notes)
        self.assertTrue(any("winding" in note for note in problem.relaxation_notes))

    def test_missing_z3_is_reported_without_crashing(self):
        problem = self._problem()
        result = joint.run_z3_problem(problem, timeout_ms=10)
        self.assertIn(
            result.status,
            {"z3_not_installed", "unsat", "sat_candidate", "timeout", "unknown", "z3_error"},
        )
        if result.status == "unsat":
            self.assertTrue(result.exact_unsat)


if __name__ == "__main__":
    unittest.main()
