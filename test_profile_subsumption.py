import unittest

import profile_subsumption as subsumption
import solution_canonicalization as canonical
import symbolic_enumerator as base


def literal(name, inverse=False):
    return base.Literal(name, inverse)


def point(name, sign=1, *, zero=False, pole=None):
    return canonical.PointDecoration(
        class_id=name,
        sign=0 if zero else sign,
        fixed_zero=zero,
        pole=pole,
    )


def solution(segments, points):
    return canonical.DecoratedSolution(
        segments=tuple(segments),
        points=tuple(points),
        mappings=(),
    )


class ProfileSubsumptionTests(unittest.TestCase):
    def test_curve_variable_can_expand_to_decorated_nonempty_path(self):
        general = solution(
            [literal("X"), literal("X", True)],
            [point("p0", pole="P0"), point("p1", pole="P1")],
        )
        refined = solution(
            [literal("A"), literal("B"), literal("B", True), literal("A", True)],
            [
                point("q0", pole="P0"),
                point("beta"),
                point("q1", pole="P1"),
                point("beta", -1),
            ],
        )
        certificate = subsumption.find_shape_subsumption(general, refined)
        self.assertIsNotNone(certificate)
        assert certificate is not None
        self.assertEqual(certificate.variable_substitution["V0"].text, "V0 (a1) V1")
        self.assertEqual(certificate.scope, "contour_shape_family")
        self.assertFalse(certificate.copy_mappings_checked_exactly)

    def test_nonuniform_refinement_is_not_a_variable_substitution(self):
        general = solution(
            [literal("X"), literal("X", True), literal("X")],
            [
                point("p0", pole="P0"),
                point("a"),
                point("p1", pole="P1"),
            ],
        )
        refined = solution(
            [literal("A"), literal("A", True), literal("B"), literal("B", True)],
            [
                point("q0", pole="P0"),
                point("b"),
                point("q1", pole="P1"),
                point("b", -1),
            ],
        )
        self.assertIsNone(subsumption.find_shape_subsumption(general, refined))

    def test_reduction_keeps_nonfree_general_profiles(self):
        general = solution(
            [literal("X"), literal("X", True)],
            [point("p0", pole="P0"), point("p1", pole="P1")],
        )
        refined = solution(
            [literal("A"), literal("B"), literal("B", True), literal("A", True)],
            [
                point("q0", pole="P0"),
                point("beta"),
                point("q1", pole="P1"),
                point("beta", -1),
            ],
        )
        g_can = canonical.canonicalize_decorated_data(general)
        r_can = canonical.canonicalize_decorated_data(refined)
        retained, absorbed = subsumption.reduce_profiles(
            [
                subsumption.ProfileReductionEntry(
                    1, general, g_can.key, g_can.canonical_json, False
                ),
                subsumption.ProfileReductionEntry(
                    2, refined, r_can.key, r_can.canonical_json, True
                ),
            ]
        )
        self.assertEqual(set(retained), {1, 2})
        self.assertFalse(absorbed)


if __name__ == "__main__":
    unittest.main()
