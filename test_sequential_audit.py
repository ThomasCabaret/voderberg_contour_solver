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
            collect_profiles=False,
            run_z3=False,
            show_progress=False,
            parity_diagnostics=False,
        )
        self.assertIn("pipeline_sequence", result)
        self.assertIn("final_survivors_after_z3_unsat_rejections", result["experimental_pipeline_counts"])
        self.assertEqual(
            result["experimental_pipeline_counts"]["z3_solver_invocations"],
            0,
        )

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
