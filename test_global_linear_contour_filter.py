import unittest
from dataclasses import replace
from fractions import Fraction

import external_boundary_constraints as external
import global_linear_contour_filter as global_linear
import joint_translation_z3
import pole_angle_filter as poles
import symbolic_enumerator as base


class GlobalLinearContourFilterTests(unittest.TestCase):
    @staticmethod
    def _empty_poles():
        return poles.PoleAngleAnalysis(
            constraints=(),
            joint_capacity_pi_units=Fraction(0),
            feasible=True,
            discard_reason=None,
        )

    @staticmethod
    def _boundary(name, variables, turn):
        segments = tuple(
            external.BoundarySegment(
                copy=external.REFERENCE_COPY,
                mirror_sign=base.DIRECT,
                literal=base.Literal(variable, False),
                occurrence_index=index,
            )
            for index, variable in enumerate(variables)
        )
        point = external.BoundaryPoint(
            physical_point=f"{name}:P",
            source_points=(f"{name}:P",),
            turn=turn,
            kind="test",
        )
        internal_points = tuple(
            external.BoundaryPoint(
                physical_point=f"{name}:Q{index}",
                source_points=(f"{name}:Q{index}",),
                turn=external.AngleForm.zero(),
                kind="test",
            )
            for index in range(max(0, len(segments) - 1))
        )
        return external.BoundaryPath(
            name=name,
            segments=segments,
            points=(point, *internal_points, point),
            initial_words=(),
        )

    @staticmethod
    def _system(inner, outer, equations=()):
        return external.JointBoundarySystem(
            inner_boundary=inner,
            outer_boundary=outer,
            curve_turn_solution=None,
            rotation_equations=tuple(equations),
            translation_equations=(),
            rotation_analysis=None,
            translation_analysis=None,
        )

    def test_voderberg_global_linear_system_is_feasible(self):
        case, state = joint_translation_z3.find_voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        pole_analysis = poles.analyze_pole_angles(case, state)
        analysis = global_linear.analyze_global_linear_contours(
            system, pole_analysis
        )
        self.assertTrue(analysis.feasible)
        self.assertGreater(analysis.strict_margin.numerator, 0)
        self.assertEqual(
            dict(analysis.inner_perimeter_coefficients),
            dict(analysis.outer_perimeter_coefficients),
        )

    def test_composite_external_turn_must_be_principal(self):
        theta = "Theta0"
        inner = self._boundary(
            "inner", ("W0",), external.AngleForm.from_mapping({theta: 1})
        )
        outer = self._boundary(
            "outer", ("W0",), external.AngleForm(pi_constant=Fraction(-1))
        )
        equations = (
            external.RotationEquation(
                boundary="inner",
                lhs=external.AngleForm.from_mapping({theta: 1}),
                target_pi=Fraction(0),
            ),
            external.RotationEquation(
                boundary="outer",
                lhs=external.AngleForm.from_mapping({theta: 1}),
                target_pi=Fraction(0),
            ),
        )
        analysis = global_linear.analyze_global_linear_contours(
            self._system(inner, outer, equations), self._empty_poles()
        )
        self.assertFalse(analysis.feasible)
        self.assertEqual(analysis.status, "angular_block_reject")
        self.assertEqual(
            analysis.angle_block.status,
            "only_degenerate_boundary_turns_feasible",
        )

    def test_inner_and_outer_perimeters_are_solved_together(self):
        inner = self._boundary("inner", ("W0",), external.AngleForm.zero())
        outer = self._boundary(
            "outer", ("W0", "W0"), external.AngleForm.zero()
        )
        equations = (
            external.RotationEquation(
                boundary="inner",
                lhs=external.AngleForm.zero(),
                target_pi=Fraction(0),
            ),
            external.RotationEquation(
                boundary="outer",
                lhs=external.AngleForm.zero(),
                target_pi=Fraction(0),
            ),
        )
        analysis = global_linear.analyze_global_linear_contours(
            self._system(inner, outer, equations), self._empty_poles()
        )
        self.assertFalse(analysis.feasible)
        self.assertEqual(analysis.status, "length_block_reject")
        self.assertEqual(
            analysis.length_block.status,
            "incompatible_inner_outer_perimeters",
        )

    def test_length_block_accepts_different_words_with_compatible_positive_lengths(self):
        inner = self._boundary(
            "inner", ("W0", "W1"), external.AngleForm.zero()
        )
        outer = self._boundary(
            "outer", ("W0", "W0", "W1"), external.AngleForm.zero()
        )
        equations = (
            external.RotationEquation(
                boundary="inner",
                lhs=external.AngleForm.zero(),
                target_pi=Fraction(0),
            ),
            external.RotationEquation(
                boundary="outer",
                lhs=external.AngleForm.zero(),
                target_pi=Fraction(0),
            ),
        )
        analysis = global_linear.analyze_global_linear_contours(
            self._system(inner, outer, equations), self._empty_poles()
        )
        # L0 + L1 = 1 and 2*L0 + L1 = 1 force L0 = 0.
        self.assertFalse(analysis.feasible)
        self.assertEqual(
            analysis.length_block.status,
            "only_zero_length_boundary_feasible",
        )


if __name__ == "__main__":
    unittest.main()
