import unittest

import analysis_pipeline as pipeline
import results_export
import symbolic_enumerator as base


class ResultsExportTests(unittest.TestCase):
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

    def test_voderberg_record_exposes_mapping_profile_and_status(self):
        case = self.find_voderberg_case()
        records = []
        for state, derivation in base.enumerate_terminal_states(case):
            analysis = pipeline.analyze_terminal_profile(case, state)
            record = results_export.detailed_profile_record(
                len(records), case, state, derivation, analysis
            )
            records.append(record)

        expected_profile = (
            "(P0 = a0) V0 (-a1) V1 (a2 = 0) V1^-1 (a1) V0^-1 "
            "(P1 = a3) V0 (-a1) V1 (a2 = 0) V1^-1"
        )
        exact = [record for record in records if record["solution"]["profile"] == expected_profile]
        self.assertEqual(len(exact), 1)
        record = exact[0]
        self.assertFalse(record["mapping"]["A"]["flipped"])
        self.assertFalse(record["mapping"]["B"]["flipped"])
        self.assertTrue(record["status"]["retained"])
        self.assertIn("P1 -> P0", record["mapping"]["A"]["display"])
        self.assertIn("inside A", record["mapping"]["B"]["display"])
        self.assertEqual(
            record["solution"]["profile"],
            "(P0 = a0) V0 (-a1) V1 (a2 = 0) V1^-1 "
            "(a1) V0^-1 (P1 = a3) V0 (-a1) V1 "
            "(a2 = 0) V1^-1",
        )
        self.assertEqual(record["solution"]["curve_parameters"], ["V0", "V1"])
        self.assertEqual(record["solution"]["angle_parameters"], ["a0", "a1", "a3"])


if __name__ == "__main__":
    unittest.main()
