import unittest
from dataclasses import replace

import analysis_pipeline as pipeline
import projection_topology as topology
import symbolic_enumerator as base


class ProjectionTopologyTests(unittest.TestCase):
    def test_all_placement_cases_embed_opposite_side_parity(self):
        cases = list(base.enumerate_placement_cases())
        self.assertEqual(len(cases), 2816)
        for case in cases:
            self.assertEqual(
                case.a_mirror_sign,
                topology.required_mirror_sign(case.a_direction),
            )
            self.assertEqual(
                case.b_mirror_sign,
                topology.required_mirror_sign(case.b_direction),
            )
            self.assertTrue(
                topology.is_opposite_side_contact(
                    case.a_direction, case.a_mirror_sign
                )
            )
            self.assertTrue(
                topology.is_opposite_side_contact(
                    case.b_direction, case.b_mirror_sign
                )
            )

    def test_direct_only_mode_prunes_at_placement_generation(self):
        cases = list(base.enumerate_placement_cases(allow_reflections=False))
        self.assertEqual(len(cases), 704)
        self.assertTrue(all(case.parity_label == "DD" for case in cases))
        self.assertTrue(
            all(
                case.a_direction == base.REVERSE
                and case.b_direction == base.REVERSE
                for case in cases
            )
        )

    def test_placement_rejects_an_inconsistent_embedded_parity(self):
        case = next(base.enumerate_placement_cases())
        with self.assertRaises(ValueError):
            replace(case, a_mirror_sign=-case.a_mirror_sign)

    def test_pipeline_defaults_to_placement_parity(self):
        case = None
        state = None
        for candidate in base.enumerate_placement_cases():
            if not candidate.requires_reflection:
                continue
            terminal = next(
                base.enumerate_terminal_states(
                    candidate, max_depth=10, max_states=1000
                ),
                None,
            )
            if terminal is not None:
                case = candidate
                state, _ = terminal
                break
        self.assertIsNotNone(case)
        self.assertIsNotNone(state)
        assert case is not None and state is not None
        implicit = pipeline.analyze_terminal_profile(case, state)
        explicit = pipeline.analyze_terminal_profile(
            case,
            state,
            mirror_sign_a=case.a_mirror_sign,
            mirror_sign_b=case.b_mirror_sign,
        )
        self.assertEqual(implicit.angle_solution, explicit.angle_solution)
        self.assertEqual(implicit.total_turn, explicit.total_turn)
        self.assertEqual(implicit.pole_angles, explicit.pole_angles)
        self.assertEqual(implicit.se2_holonomy, explicit.se2_holonomy)


if __name__ == "__main__":
    unittest.main()
