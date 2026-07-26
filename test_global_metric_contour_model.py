import unittest

import external_boundary_constraints as external
import global_metric_contour_model as metric
import symbolic_enumerator as base


class GlobalMetricContourModelTests(unittest.TestCase):
    @staticmethod
    def _boundary():
        literals = (
            base.Literal("W0", False),
            base.Literal("W0", True),
            base.Literal("W1", False),
            base.Literal("W1", True),
        )
        mirrors = (base.DIRECT, base.DIRECT, base.REFLECTED, base.REFLECTED)
        segments = tuple(
            external.BoundarySegment(
                copy="test",
                mirror_sign=mirror,
                literal=literal,
                occurrence_index=index,
            )
            for index, (literal, mirror) in enumerate(zip(literals, mirrors))
        )
        points = tuple(
            external.BoundaryPoint(
                physical_point="P" if index in (0, len(segments)) else f"Q{index}",
                source_points=("P",),
                turn=external.AngleForm.zero(),
                kind="test",
            )
            for index in range(len(segments) + 1)
        )
        return external.BoundaryPath(
            name="test_boundary",
            segments=segments,
            points=points,
            initial_words=(),
        )

    @staticmethod
    def _system(boundary):
        curve_turns = external.CurveTurnSolution(
            equations=(), assignments=(), zero_variables=()
        )
        return external.JointBoundarySystem(
            inner_boundary=boundary,
            outer_boundary=boundary,
            curve_turn_solution=curve_turns,
            rotation_equations=(),
            translation_equations=(),
            rotation_analysis=None,
            translation_analysis=None,
        )

    def test_reversal_and_reflection_area_signs(self):
        model = metric.build_global_metric_contour_model(
            self._system(self._boundary())
        )
        signs = tuple(
            segment.signed_arc_area_sign
            for segment in model.inner_boundary.segments
        )
        self.assertEqual(signs, (1, -1, -1, 1))

    def test_perimeter_coefficients_share_curve_lengths(self):
        model = metric.build_global_metric_contour_model(
            self._system(self._boundary())
        )
        self.assertEqual(
            dict(model.inner_boundary.perimeter_coefficients),
            {"W0": 2, "W1": 2},
        )
        self.assertEqual(model.curve_variables, ("W0", "W1"))


if __name__ == "__main__":
    unittest.main()
