import unittest

import analysis_pipeline as pipeline
import profile_formatter
import symbolic_enumerator as base
import angle_constraints as angles


class ProfileFormatterTests(unittest.TestCase):
    @staticmethod
    def find_voderberg_case():
        expected_loci = {
            "A_start": "P1",
            "A_end": "P0",
            "B_start": "P1",
            "B_end": "A",
        }
        matches = [
            case
            for case in base.enumerate_placement_cases()
            if case.marker_locus_map() == expected_loci
            and case.a_interior_blocks == (("B_end",),)
            and case.b_interior_blocks == ()
            and case.a_direction == base.REVERSE
            and case.b_direction == base.REVERSE
        ]
        if len(matches) != 1:
            raise AssertionError(f"Expected one Voderberg case, found {len(matches)}")
        return matches[0]

    def test_voderberg_profile_interleaves_curve_and_angle_classes(self):
        case = self.find_voderberg_case()
        for state, _derivation in base.enumerate_terminal_states(case):
            a_text, b_text = angles.state_profile_text(case, state)
            if (
                a_text == "V0 V1 V1^-1 V0^-1"
                and b_text == "V0 V1 V1^-1"
            ):
                analysis = pipeline.analyze_terminal_profile(case, state)
                profile = profile_formatter.build_formal_profile(
                    case, state, analysis.angle_solution
                )
                self.assertEqual(
                    profile.text,
                    "(P0 = a0) V0 (-a1) V1 (a2 = 0) V1^-1 "
                    "(a1) V0^-1 (P1 = a3) V0 (-a1) V1 "
                    "(a2 = 0) V1^-1",
                )
                self.assertEqual(profile.curve_parameters, ("V0", "V1"))
                self.assertEqual(profile.free_angle_parameters, ("a0", "a1", "a3"))
                self.assertEqual(profile.fixed_zero_angle_classes, ("a2",))
                return
        self.fail("Exact Voderberg terminal profile was not found")

    def test_named_points_keep_their_names(self):
        case = self.find_voderberg_case()
        state, _derivation = next(base.enumerate_terminal_states(case))
        analysis = pipeline.analyze_terminal_profile(case, state)
        profile = profile_formatter.build_formal_profile(
            case, state, analysis.angle_solution
        )
        self.assertTrue(profile.text.startswith("(P0 = "))
        self.assertIn("(P1 = ", profile.text)


if __name__ == "__main__":
    unittest.main()
