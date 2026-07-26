import unittest

import external_boundary_constraints as external
import joint_translation_z3
import placed_copy_geometry as placed


class PlacedCopyGeometryTests(unittest.TestCase):
    def test_voderberg_builds_three_copies_in_one_frame(self):
        case, state = joint_translation_z3.find_voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        analysis = placed.analyze_placed_copy_geometry(case, state, system)
        copies = {point.copy for point in analysis.points}
        self.assertEqual(copies, {placed.REFERENCE, placed.COPY_A, placed.COPY_B})
        self.assertGreater(len(analysis.contact_point_equations), 4)
        self.assertTrue(analysis.passes_filter, analysis.discard_reason)

    def test_every_contact_endpoint_is_exported_for_polynomial_backend(self):
        case, state = joint_translation_z3.find_voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        analysis = placed.analyze_placed_copy_geometry(case, state, system)
        projections = {item.projection for item in analysis.contact_point_equations}
        self.assertEqual(projections, {"A", "B"})
        for equation in analysis.contact_point_equations:
            self.assertTrue(equation.reference_label.startswith("reference:"))
            self.assertIn(equation.copy_label.split(":", 1)[0], {"copy_A", "copy_B"})

    def test_scope_does_not_claim_generic_crossing_detection(self):
        case, state = joint_translation_z3.find_voderberg_case_and_state()
        system = external.build_joint_boundary_system(case, state)
        analysis = placed.analyze_placed_copy_geometry(case, state, system)
        payload = analysis.to_dict()
        self.assertFalse(payload["scope"]["detects_generic_curved_arc_crossings"])
        self.assertFalse(payload["scope"]["rejects_extra_cross_copy_point_contacts"])


if __name__ == "__main__":
    unittest.main()
