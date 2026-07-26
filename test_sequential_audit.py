import json
import tempfile
import unittest
from pathlib import Path

import audit_geometric_filters as audit


class SequentialAuditTests(unittest.TestCase):
    def test_small_audit_writes_sequential_counts(self):
        result = audit.audit(
            1,
            12,
            formal_solver_mode="legacy-bounded",
            collect_profiles=False,
            run_z3=False,
            show_progress=False,
            parity_diagnostics=False,
        )
        self.assertIn("pipeline_sequence", result)
        counts = result["experimental_pipeline_counts"]
        self.assertIn("exact_joint_angle_rejections", counts)
        self.assertIn("exact_placed_copy_geometry_rejections", counts)
        self.assertIn("final_survivors_after_z3_unsat_rejections", counts)
        self.assertEqual(
            result["experimental_pipeline_counts"]["z3_solver_invocations"],
            0,
        )

    def test_type_selection_is_applied_after_formal_solving(self):
        result = audit.audit(
            5,
            100,
            formal_solver_mode="legacy-bounded",
            max_cycle_unrolls=3,
            canonicalize_solutions=False,
            collect_profiles=False,
            collect_survivors=False,
            run_z3=False,
            show_progress=False,
            parity_diagnostics=False,
            voderberg_type_selection="type2",
        )
        summary = result["voderberg_type_classification_summary"]
        formal_summary = result["formal_equation_audit_summary"]
        self.assertTrue(formal_summary["positive_length_filter_enabled"])
        self.assertEqual(formal_summary["positive_length_infeasible_case_count_detected"], 2184)
        self.assertEqual(formal_summary["positive_length_filter_rejected_case_count"], 2184)
        self.assertEqual(
            formal_summary["additional_positive_length_rejections_before_branching_case_count"],
            1640,
        )
        self.assertEqual(formal_summary["submitted_to_branching_solver_case_count"], 632)
        self.assertEqual(summary["terminal_profile_count_before_type_selection"], 1078)
        self.assertEqual(summary["type2_compatible_profile_count"], 58)
        self.assertEqual(summary["terminal_profile_count_selected_for_downstream_pipeline"], 58)
        self.assertEqual(summary["type2_compatible_placement_case_count_generated"], 100)
        self.assertEqual(
            result["physical_pipeline_counts"][
                "formal_terminal_profile_instances_selected_for_downstream_pipeline"
            ],
            58,
        )

    def test_terminal_mapping_is_exported_without_solution_canonicalization(self):
        result = audit.audit(
            5,
            100,
            formal_solver_mode="legacy-bounded",
            canonicalize_solutions=False,
            collect_profiles=False,
            collect_survivors=True,
            run_z3=False,
            show_progress=False,
            parity_diagnostics=False,
        )
        self.assertTrue(result["survivors"])
        mapping = result["survivors"][0]["terminal_mapping"]
        self.assertEqual(mapping["schema_version"], "terminal-contact-mapping-v1")
        self.assertEqual(len(mapping["mappings"]), 2)
        self.assertTrue(mapping["mappings"][0]["segment_pairs"])

    def test_zero_cli_limit_means_unbounded(self):
        self.assertIsNone(audit._normalize_optional_limit(0, "--max-depth"))
        self.assertEqual(audit._normalize_optional_limit(7, "--max-depth"), 7)
        with self.assertRaises(ValueError):
            audit._normalize_optional_limit(-1, "--max-depth")

    def test_atomic_json_streaming_round_trip(self):
        payload = {"profiles": [{"profile_id": index, "text": "x" * 50} for index in range(500)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            audit._atomic_json(path, payload, indent=None)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded["profiles"]), 500)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
