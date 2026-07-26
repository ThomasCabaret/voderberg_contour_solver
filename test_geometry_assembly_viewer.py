import json
import tempfile
import unittest
from pathlib import Path

import geometry_assembly_viewer as viewer


class GeometryAssemblyViewerTests(unittest.TestCase):
    def _files(self, directory: Path, *, length_mismatch: bool = False) -> tuple[Path, Path]:
        top_y = 1.5 if length_mismatch else 1.0
        candidate = {
            "metadata": {},
            "candidates": [{
                "profile_id": 7,
                "case_id": 11,
                "formal_profile": "demo profile",
                "vertices": [[0, 0], [2, 0], [2, top_y], [0, 1], [0, 0]],
                "edge_metadata": [
                    {"occurrence": 0, "subedge_index": 0, "start_vertex": 0, "end_vertex": 1},
                    {"occurrence": 1, "subedge_index": 0, "start_vertex": 1, "end_vertex": 2},
                    {"occurrence": 2, "subedge_index": 0, "start_vertex": 2, "end_vertex": 3},
                    {"occurrence": 3, "subedge_index": 0, "start_vertex": 3, "end_vertex": 0},
                ],
            }],
        }
        survivors = {
            "metadata": {},
            "profiles": [{
                "profile_id": 7,
                "case_id": 11,
                "solution": {"profile": "demo profile"},
                "terminal_mapping": {
                    "mappings": [
                        {
                            "copy_index": 0,
                            "mirror_sign": 1,
                            "source_start_boundary": 0,
                            "source_end_boundary": 1,
                            "target_start_boundary": 2,
                            "target_end_boundary": 3,
                            "segment_pairs": [{"source_position": 99, "target_position": 98}],
                        },
                        {
                            "copy_index": 1,
                            "mirror_sign": -1,
                            "source_start_boundary": 1,
                            "source_end_boundary": 2,
                            "target_start_boundary": 2,
                            "target_end_boundary": 1,
                            "segment_pairs": [],
                        },
                    ]
                },
            }],
        }
        candidate_path = directory / "geometric_candidates.json"
        survivor_path = directory / "geometric_filter_survivors.json"
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        survivor_path.write_text(json.dumps(survivors), encoding="utf-8")
        return candidate_path, survivor_path

    def test_uses_boundary_metadata_not_segment_pairs(self):
        with tempfile.TemporaryDirectory() as name:
            candidate_path, survivor_path = self._files(Path(name))
            result = viewer.assemble_candidates(candidate_path, survivor_path)
        item = result[0]
        self.assertEqual(len(item.placements), 2)
        self.assertTrue(all(placement.ok for placement in item.placements))

    def test_direct_transform_maps_target_pair_to_source_pair(self):
        with tempfile.TemporaryDirectory() as name:
            candidate_path, survivor_path = self._files(Path(name))
            item = viewer.assemble_candidates(candidate_path, survivor_path)[0]
        transform = item.placements[0].transform
        assert transform is not None
        self.assertAlmostEqual(transform.apply((2, 1))[0], 0.0)
        self.assertAlmostEqual(transform.apply((2, 1))[1], 0.0)
        self.assertAlmostEqual(transform.apply((0, 1))[0], 2.0)
        self.assertAlmostEqual(transform.apply((0, 1))[1], 0.0)

    def test_reflected_transform_uses_prescribed_parity(self):
        with tempfile.TemporaryDirectory() as name:
            candidate_path, survivor_path = self._files(Path(name))
            item = viewer.assemble_candidates(candidate_path, survivor_path)[0]
        transform = item.placements[1].transform
        assert transform is not None
        self.assertTrue(transform.mirror)
        self.assertAlmostEqual(transform.apply((2, 1))[0], 2.0)
        self.assertAlmostEqual(transform.apply((2, 1))[1], 0.0)
        self.assertAlmostEqual(transform.apply((2, 0))[0], 2.0)
        self.assertAlmostEqual(transform.apply((2, 0))[1], 1.0)

    def test_chord_length_mismatch_marks_only_copy_as_failed(self):
        with tempfile.TemporaryDirectory() as name:
            candidate_path, survivor_path = self._files(Path(name), length_mismatch=True)
            item = viewer.assemble_candidates(candidate_path, survivor_path)[0]
        self.assertFalse(item.placements[0].ok)
        self.assertIsNotNone(item.placements[0].error)

    def test_case_mismatch_does_not_search_for_another_mapping(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            candidate_path, survivor_path = self._files(directory)
            payload = json.loads(survivor_path.read_text(encoding="utf-8"))
            payload["profiles"][0]["case_id"] = 12
            survivor_path.write_text(json.dumps(payload), encoding="utf-8")
            item = viewer.assemble_candidates(candidate_path, survivor_path)[0]
        self.assertIsNotNone(item.link_error)
        self.assertFalse(item.placements)

    def test_duplicate_profile_id_is_reported_not_ranked(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            candidate_path, survivor_path = self._files(directory)
            payload = json.loads(survivor_path.read_text(encoding="utf-8"))
            payload["profiles"].append(dict(payload["profiles"][0]))
            survivor_path.write_text(json.dumps(payload), encoding="utf-8")
            item = viewer.assemble_candidates(candidate_path, survivor_path)[0]
        self.assertIsNotNone(item.link_error)
        self.assertIn("2 records", item.link_error)


if __name__ == "__main__":
    unittest.main()
