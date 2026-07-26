import json
import tempfile
import unittest
from pathlib import Path

import results_web
import settings


class ResultsWebTests(unittest.TestCase):
    def test_loads_detailed_profile_payload(self):
        payload = {
            "metadata": {"schema_version": settings.WEB_SCHEMA_VERSION, "profile_count": 1},
            "summary": {},
            "profiles": [{"profile_id": 0, "solution": {"profile": "(P0 = a0) V0 (P1 = a1) V1", "formal_profile": {"angle_classes": []}}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = results_web.load_payload(path)
        self.assertEqual(loaded["profiles"][0]["profile_id"], 0)
        self.assertTrue(loaded["metadata"]["source_file"].endswith("profiles.json"))

    def test_rejects_summary_only_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({"examples": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                results_web.load_payload(path)

    def test_html_contains_requested_controls(self):
        html = results_web.HTML_PAGE
        self.assertIn("Appariement retenu", html)
        self.assertIn("Profil formel complet", html)
        self.assertIn("sortKey", html)
        self.assertIn("flipFilter", html)
        self.assertIn("voderbergTypeFilter", html)
        self.assertIn("Couche experimentale", html)
        self.assertIn("solution.profile", html)
        self.assertIn("Classes d angles", html)
        self.assertIn("previousPage", html)
        self.assertIn("pageSize", html)

    def test_default_web_file_is_survivor_export(self):
        self.assertEqual(
            results_web.DEFAULT_RESULTS_FILE.name,
            settings.AUDIT_SURVIVORS_FILENAME,
        )


if __name__ == "__main__":
    unittest.main()
