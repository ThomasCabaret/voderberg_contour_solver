import unittest

import external_boundary_constraints as external
import joint_translation_z3 as joint
import placed_copy_geometry as placed


class JointTranslationZ3Tests(unittest.TestCase):
    def _problem(self):
        case, state = joint.find_voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        placed_analysis = placed.analyze_placed_copy_geometry(case, state, system)
        return joint.build_z3_problem(
            system, placed_geometry_analysis=placed_analysis
        )

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

    def test_global_copy_isometries_are_enforced_pointwise(self):
        problem = self._problem()
        self.assertTrue(problem.global_isometry_enforced)
        self.assertGreater(problem.contact_point_equation_count, 0)
        self.assertGreater(problem.distinguished_point_inequality_count, 0)
        self.assertIn("One shared direct/reflected isometry per copy", problem.script_smt2)


    def test_metric_and_signed_area_layers_are_enabled_by_default(self):
        problem = self._problem()
        self.assertTrue(problem.metric_length_constraints_enabled)
        self.assertTrue(problem.signed_area_constraints_enabled)
        self.assertTrue(problem.length_symbol_map)
        self.assertTrue(problem.arc_area_symbol_map)
        self.assertIn("Chord/length layer", problem.script_smt2)
        self.assertIn("Signed-area layer", problem.script_smt2)
        self.assertIn("external union contains three congruent", problem.script_smt2)
        self.assertGreater(problem.metric_constraint_count, 0)
        self.assertGreater(problem.signed_area_constraint_count, 0)

    def test_polynomial_layers_are_independently_configurable(self):
        case, state = joint.find_voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        placed_analysis = placed.analyze_placed_copy_geometry(case, state, system)
        metric_only = joint.build_z3_problem(
            system,
            placed_geometry_analysis=placed_analysis,
            enable_metric_lengths=True,
            enable_signed_areas=False,
        )
        self.assertTrue(metric_only.metric_length_constraints_enabled)
        self.assertFalse(metric_only.signed_area_constraints_enabled)
        self.assertTrue(metric_only.length_symbol_map)
        self.assertFalse(metric_only.arc_area_symbol_map)

        closure_only = joint.build_z3_problem(
            system,
            placed_geometry_analysis=placed_analysis,
            enable_metric_lengths=False,
            enable_signed_areas=False,
        )
        self.assertFalse(closure_only.metric_length_constraints_enabled)
        self.assertFalse(closure_only.signed_area_constraints_enabled)
        self.assertFalse(closure_only.length_symbol_map)
        self.assertIn("Homogeneous normalization", closure_only.script_smt2)

    def test_signed_area_requires_metric_layer(self):
        case, state = joint.find_voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        with self.assertRaises(ValueError):
            joint.build_z3_problem(
                system,
                enable_metric_lengths=False,
                enable_signed_areas=True,
            )

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
