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

    def test_reflected_self_contact_removes_artificial_intermediate_points(self):
        profile = geometry.SearchProfile(
            profile_id=0,
            case_id=0,
            formal_text="test",
            occurrences=(geometry.SegmentOccurrence("V0", False),),
            point_expressions=("0",),
            free_angles=(),
            curve_variables=("V0",),
            kappa_assignments={"V0": "K0"},
            terminal_mapping={
                "schema_version": "terminal-contact-mapping-v1",
                "segment_count": 1,
                "mappings": [{
                    "copy_index": 0,
                    "mirror_sign": -1,
                    "segment_pairs": [{
                        "source_position": 0,
                        "source_orientation": 1,
                        "target_position": 0,
                        "target_orientation": 1,
                    }],
                }],
            },
        )
        plan = geometry._template_plan(
            profile,
            3,
            enforce_contact_template_constraints=True,
        )
        self.assertEqual(plan.component_for("V0").effective_edge_count, 1)
        self.assertEqual(len(geometry._bounds(profile, 3, plan)), 1)
        vertices, metadata, *_ = geometry._simulate(profile, [1.25], 3, plan)
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(metadata), 1)

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


class FormalCurveTermGeometryIntegrationTests(unittest.TestCase):
    def _profile(self, *, target_orientation, mirror_sign, exported=True):
        occurrence = geometry.SegmentOccurrence("V0", False)
        mapping = {
            "schema_version": "terminal-contact-mapping-v1",
            "segment_count": 1,
            "mappings": [{
                "copy_index": 0,
                "mirror_sign": mirror_sign,
                "segment_pairs": [{
                    "source_position": 0,
                    "source_orientation": 1,
                    "target_position": 0,
                    "target_orientation": target_orientation,
                }],
            }],
        }
        formal = geometry.curve_term_solver.solve_curve_terms(
            curve_variables=("V0",),
            occurrences=(occurrence,),
            terminal_mapping=mapping,
        )
        return geometry.SearchProfile(
            profile_id=999,
            case_id=999,
            formal_text="(P0) V0",
            occurrences=(occurrence,),
            point_expressions=("0",),
            free_angles=(),
            curve_variables=("V0",),
            kappa_assignments={"V0": "K0"},
            terminal_mapping=mapping,
            curve_term_solution=(formal.to_dict() if exported else None),
        )

    def test_straight_formal_value_removes_requested_intermediate_points(self):
        profile = self._profile(target_orientation=1, mirror_sign=-1)
        plan = geometry._template_plan(
            profile,
            intermediate_points=5,
            enforce_contact_template_constraints=True,
        )
        component = plan.component_for("V0")
        self.assertEqual(component.mode, "straight")
        self.assertEqual(component.requested_edge_count, 6)
        self.assertEqual(component.effective_edge_count, 1)
        self.assertEqual(component.free_turn_count, 0)

    def test_mirror_symmetric_term_uses_one_shared_half_template(self):
        profile = self._profile(target_orientation=-1, mirror_sign=-1)
        plan = geometry._template_plan(
            profile,
            intermediate_points=3,
            enforce_contact_template_constraints=True,
        )
        component = plan.component_for("V0")
        self.assertEqual(component.mode, "endpoint_swapping_reflection")
        self.assertEqual(component.effective_edge_count, 4)
        self.assertLess(component.free_length_count, component.effective_edge_count)

    def test_stale_exported_curve_terms_are_rejected(self):
        profile = self._profile(target_orientation=1, mirror_sign=-1)
        stale = dict(profile.curve_term_solution)
        stale["terms"] = {"V0": {"kind": "curve_parameter", "text": "C0"}}
        profile = geometry.SearchProfile(**{
            **profile.__dict__,
            "curve_term_solution": stale,
        })
        with self.assertRaisesRegex(ValueError, "stale curve terms"):
            geometry._template_plan(
                profile,
                intermediate_points=3,
                enforce_contact_template_constraints=True,
            )


if __name__ == "__main__":
    unittest.main()
