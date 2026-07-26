import unittest
from fractions import Fraction

import external_boundary_constraints as external
import joint_angle_feasibility as joint
import joint_translation_z3
import pole_angle_filter as poles
import turning_filter


class JointAngleFeasibilityTests(unittest.TestCase):
    @staticmethod
    def _pole_analysis(first, second):
        constraints = (
            poles.PoleConstraint(
                pole="P0",
                contact_points=("p0", "p1", "p2"),
                turn_coefficients=tuple(
                    poles.IntegerCoefficient(name, value)
                    for name, value in sorted(first.items())
                ),
                individual_capacity_pi_units=sum(abs(value) for value in first.values()),
            ),
            poles.PoleConstraint(
                pole="P1",
                contact_points=("q0", "q1", "q2"),
                turn_coefficients=tuple(
                    poles.IntegerCoefficient(name, value)
                    for name, value in sorted(second.items())
                ),
                individual_capacity_pi_units=sum(abs(value) for value in second.values()),
            ),
        )
        return poles.PoleAngleAnalysis(
            constraints=constraints,
            joint_capacity_pi_units=Fraction(2),
            feasible=True,
            discard_reason=None,
        )

    def test_voderberg_complete_angular_system_is_feasible(self):
        case, state = joint_translation_z3.find_voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        pole_analysis = poles.analyze_pole_angles(case, state)
        analysis = joint.analyze_joint_angle_feasibility(
            system.rotation_equations, pole_analysis
        )
        self.assertTrue(analysis.feasible)
        self.assertGreater(analysis.strict_margin.numerator, 0)

    def test_separately_feasible_constraints_can_be_jointly_infeasible(self):
        theta = "Theta0"
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
        pole_analysis = self._pole_analysis({theta: 1}, {theta: 1})
        analysis = joint.analyze_joint_angle_feasibility(equations, pole_analysis)
        self.assertFalse(analysis.feasible)
        self.assertEqual(analysis.status, "infeasible_closed_linear_system")

    def test_boundary_only_solution_is_rejected(self):
        theta = "Theta0"
        equations = (
            external.RotationEquation(
                boundary="inner",
                lhs=external.AngleForm.from_mapping({theta: 1}),
                target_pi=Fraction(1),
            ),
            external.RotationEquation(
                boundary="outer",
                lhs=external.AngleForm.from_mapping({theta: 1}),
                target_pi=Fraction(1),
            ),
        )
        pole_analysis = self._pole_analysis({theta: 1}, {theta: 1})
        analysis = joint.analyze_joint_angle_feasibility(equations, pole_analysis)
        self.assertFalse(analysis.feasible)
        self.assertEqual(analysis.status, "only_principal_angle_boundary_feasible")
        self.assertEqual(analysis.strict_margin.numerator, 0)

    def test_redundant_rotation_equations_are_reduced_before_simplex(self):
        equations = (
            external.RotationEquation(
                boundary="inner",
                lhs=external.AngleForm.from_mapping({"Theta0": 1, "Theta1": 1}),
                target_pi=Fraction(2),
            ),
            external.RotationEquation(
                boundary="outer",
                lhs=external.AngleForm.from_mapping({"Theta0": 3, "Theta1": 3}),
                target_pi=Fraction(6),
            ),
        )
        pole_analysis = self._pole_analysis(
            {"Theta0": 1, "Theta1": 2},
            {"Theta0": 2, "Theta1": 1},
        )
        analysis = joint.analyze_joint_angle_feasibility(equations, pole_analysis)
        self.assertFalse(analysis.feasible)
        self.assertEqual(analysis.strict_margin.numerator, 0)


if __name__ == "__main__":
    unittest.main()
