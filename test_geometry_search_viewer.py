import json
import tempfile
import unittest
from pathlib import Path

import geometry_search_viewer as geometry
import settings


class GeometrySearchViewerTests(unittest.TestCase):
    def test_default_searches_all_survivors(self):
        self.assertEqual(settings.GEOMETRY_DEFAULT_MAX_PROFILES, 0)

    def test_occurrence_parser(self):
        parsed = geometry._parse_occurrences("(P0) V0 V1^-1 (P1) V0")
        self.assertEqual([item.text for item in parsed], ["V0", "V1^-1", "V0"])

    def test_segment_intersection(self):
        self.assertTrue(
            geometry._segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))
        )
        self.assertFalse(
            geometry._segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))
        )

    def test_signed_area(self):
        self.assertGreater(
            geometry._signed_area([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            0,
        )

    def test_configurable_polyline_sizes(self):
        self.assertEqual(geometry._edge_count(0), 1)
        self.assertEqual(geometry._edge_count(1), 2)
        self.assertEqual(geometry._edge_count(3), 4)
        self.assertEqual(geometry._free_internal_turn_count(0), 0)
        self.assertEqual(geometry._free_internal_turn_count(2), 1)

    def test_inverse_template_reverses_lengths_and_turns(self):
        lengths, turns = geometry._oriented_template(
            (1.0, 2.0, 3.0), (0.2, -0.5), True
        )
        self.assertEqual(lengths, (3.0, 2.0, 1.0))
        self.assertEqual(turns, (0.5, -0.2))

    def test_parameter_count_scales_with_intermediate_points(self):
        profile = geometry.SearchProfile(
            profile_id=0,
            case_id=0,
            formal_text="test",
            occurrences=(geometry.SegmentOccurrence("V0", False),),
            point_expressions=("a0",),
            free_angles=("a0",),
            curve_variables=("V0",),
            kappa_assignments={"V0": "K0"},
        )
        self.assertEqual(len(geometry._bounds(profile, 0)), 2)
        self.assertEqual(len(geometry._bounds(profile, 1)), 4)
        self.assertEqual(len(geometry._bounds(profile, 2)), 6)

    def test_simulation_uses_requested_edge_count(self):
        profile = geometry.SearchProfile(
            profile_id=0,
            case_id=0,
            formal_text="test",
            occurrences=(geometry.SegmentOccurrence("V0", False),),
            point_expressions=("0",),
            free_angles=(),
            curve_variables=("V0",),
            kappa_assignments={"V0": "0"},
        )
        values = [1.0, 1.0, 1.0, 0.2]
        vertices, metadata, *_ = geometry._simulate(profile, values, 2)
        self.assertEqual(len(vertices), 4)
        self.assertEqual(len(metadata), 3)

    def test_streaming_top_level_array_reader_handles_small_chunks(self):
        payload = {
            "metadata": {"profile_count": 3},
            "profiles": [
                {"profile_id": 1, "text": "alpha"},
                {"profile_id": 2, "text": "beta with ] and , inside"},
                {"profile_id": 3, "nested": {"value": [1, 2, 3]}},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = list(
                geometry._iter_top_level_array(path, "profiles", chunk_size=7)
            )
        self.assertEqual([record["profile_id"] for record in records], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
