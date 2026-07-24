import unittest

from forced_point_coincidence import (
    AngleForm,
    CurveOccurrence,
    PointEquality,
    VectorExpression,
    analyze_boundary_path_forced_coincidences,
    analyze_forced_point_coincidences,
)

import external_boundary_constraints as external
import symbolic_enumerator as base


ZERO = AngleForm.zero()
PI = AngleForm.from_mapping({}, pi_constant=1)


class ForcedPointCoincidenceTests(unittest.TestCase):
    def test_consecutive_points_are_checked(self):
        analysis = analyze_forced_point_coincidences(
            [CurveOccurrence("X")],
            [ZERO],
            point_labels=["left", "right"],
            zero_displacement_variables={"X"},
            allow_cycle_closure=False,
        )
        self.assertFalse(analysis.passes_filter)
        self.assertTrue(analysis.coincidence_classes[0].contains_consecutive_points)

    def test_only_final_cycle_closure_is_automatically_allowed(self):
        analysis = analyze_forced_point_coincidences(
            [CurveOccurrence("X")],
            [ZERO],
            point_labels=["P0-start", "P0-end"],
            zero_displacement_variables={"X"},
            allow_cycle_closure=True,
        )
        self.assertTrue(analysis.passes_filter)

    def test_nonfinal_return_to_start_is_detected(self):
        analysis = analyze_forced_point_coincidences(
            [
                CurveOccurrence("X"),
                CurveOccurrence("X", inverse=True),
                CurveOccurrence("Y"),
            ],
            [PI, ZERO, ZERO],
            point_labels=["P0", "q1", "q2", "q3"],
            allow_cycle_closure=True,
        )
        self.assertFalse(analysis.passes_filter)
        self.assertIn((0, 2), analysis.coincidence_classes[0].violating_pairs)

    def test_zero_turn_between_x_and_inverse_does_not_force_retrace(self):
        analysis = analyze_forced_point_coincidences(
            [CurveOccurrence("X"), CurveOccurrence("X", inverse=True)],
            [ZERO, ZERO],
            allow_cycle_closure=False,
        )
        self.assertTrue(analysis.passes_filter)
        self.assertEqual(
            analysis.positions[-1].expression.to_text(),
            "2*R(0)D[X]",
        )

    def test_explicit_equalities_are_closed_transitively(self):
        analysis = analyze_forced_point_coincidences(
            [CurveOccurrence("X"), CurveOccurrence("Y")],
            [ZERO, ZERO],
            explicit_equalities=[
                PointEquality(0, 1, "mapping one"),
                PointEquality(1, 2, "mapping two"),
            ],
            allow_cycle_closure=False,
        )
        self.assertFalse(analysis.passes_filter)
        self.assertEqual(
            analysis.coincidence_classes[0].member_indices,
            (0, 1, 2),
        )

    def test_allowed_explicit_coincidence_is_not_rejected(self):
        analysis = analyze_forced_point_coincidences(
            [CurveOccurrence("X")],
            [ZERO],
            explicit_equalities=[PointEquality(0, 1, "intended gluing")],
            allowed_coincidences=[(0, 1)],
            allow_cycle_closure=False,
        )
        self.assertTrue(analysis.passes_filter)

    def test_half_turn_phases_cancel_exactly(self):
        expression = (
            VectorExpression.zero()
            .add_chord("X", ZERO)
            .add_chord("X", PI)
        )
        self.assertTrue(expression.is_zero)

    def test_labels_do_not_merge_distinct_occurrences(self):
        analysis = analyze_forced_point_coincidences(
            [CurveOccurrence("X")],
            [ZERO],
            point_labels=["same", "same"],
            explicit_equalities=[PointEquality(0, 1)],
            allow_cycle_closure=False,
        )
        self.assertFalse(analysis.passes_filter)

    def test_boundary_path_adapter_detects_premature_return(self):
        zero = external.AngleForm.zero()
        pi = external.AngleForm(pi_constant=1)
        boundary = external.BoundaryPath(
            name="test_boundary",
            segments=(
                external.BoundarySegment("reference", base.DIRECT, base.Literal("X"), 0),
                external.BoundarySegment("reference", base.DIRECT, base.Literal("X", True), 1),
                external.BoundarySegment("reference", base.DIRECT, base.Literal("Y"), 2),
            ),
            points=(
                external.BoundaryPoint("P0", ("P0",), zero, "reference"),
                external.BoundaryPoint("q1", ("q1",), pi, "reference"),
                external.BoundaryPoint("q2", ("q2",), zero, "reference"),
                external.BoundaryPoint("P0", ("P0",), zero, "reference"),
            ),
            initial_words=(),
        )
        curve_turns = external.CurveTurnSolution(
            equations=(),
            assignments=(("X", "Kappa[X]"), ("Y", "Kappa[Y]")),
            zero_variables=(),
        )
        analysis = analyze_boundary_path_forced_coincidences(
            boundary, curve_turns
        )
        self.assertFalse(analysis.passes_filter)
        self.assertTrue(
            any((0, 2) in item.violating_pairs for item in analysis.coincidence_classes)
        )


if __name__ == "__main__":
    unittest.main()
