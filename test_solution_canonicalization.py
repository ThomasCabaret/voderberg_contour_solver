import unittest
from dataclasses import replace

import analysis_pipeline
import formal_equation_audit
import profile_formatter
import solution_canonicalization as canonical
import symbolic_enumerator as base


class DecoratedSolutionCanonicalizationTests(unittest.TestCase):
    def _solution(self):
        segments = (
            base.Literal("X"),
            base.Literal("Y"),
            base.Literal("Y", True),
            base.Literal("X", True),
            base.Literal("X"),
            base.Literal("Y"),
        )
        points = (
            canonical.PointDecoration("theta0", 1, False, "P0"),
            canonical.PointDecoration("theta1", -1, False, None),
            canonical.PointDecoration("zero", 0, True, None),
            canonical.PointDecoration("theta1", 1, False, None),
            canonical.PointDecoration("theta2", 1, False, "P1"),
            canonical.PointDecoration("theta1", -1, False, None),
        )
        mapping_a = canonical.ContactMapping(
            source_start_boundary=0,
            source_end_boundary=4,
            target_start_boundary=2,
            target_end_boundary=0,
            mirror_sign=base.DIRECT,
            pairs=(
                (canonical.DirectedSegmentRef(0, 1), canonical.DirectedSegmentRef(1, -1)),
                (canonical.DirectedSegmentRef(1, 1), canonical.DirectedSegmentRef(0, -1)),
                (canonical.DirectedSegmentRef(2, 1), canonical.DirectedSegmentRef(5, -1)),
                (canonical.DirectedSegmentRef(3, 1), canonical.DirectedSegmentRef(4, -1)),
            ),
        )
        mapping_b = canonical.ContactMapping(
            source_start_boundary=4,
            source_end_boundary=0,
            target_start_boundary=0,
            target_end_boundary=2,
            mirror_sign=base.REFLECTED,
            pairs=(
                (canonical.DirectedSegmentRef(4, 1), canonical.DirectedSegmentRef(0, 1)),
                (canonical.DirectedSegmentRef(5, 1), canonical.DirectedSegmentRef(1, 1)),
            ),
        )
        return canonical.DecoratedSolution(segments, points, (mapping_a, mapping_b))

    def test_renaming_and_copy_permutation_do_not_change_key(self):
        solution = self._solution()
        renamed_segments = tuple(
            base.Literal(
                {"X": "LongCurve", "Y": "ShortCurve"}[literal.variable],
                literal.inverse,
            )
            for literal in solution.segments
        )
        renamed_points = tuple(
            canonical.PointDecoration(
                {"theta0": "u", "theta1": "v", "theta2": "w", "zero": "z"}[point.class_id],
                point.sign,
                point.fixed_zero,
                point.pole,
            )
            for point in solution.points
        )
        renamed = canonical.DecoratedSolution(
            renamed_segments,
            renamed_points,
            tuple(reversed(solution.mappings)),
        )
        self.assertEqual(
            canonical.canonicalize_decorated_data(solution).key,
            canonical.canonicalize_decorated_data(renamed).key,
        )

    def test_global_mirror_does_not_change_key(self):
        solution = self._solution()
        pole0 = next(i for i, point in enumerate(solution.points) if point.pole == "P0")
        segments, points = canonical._transformed_cycle(solution, pole0, True)
        mappings = tuple(
            canonical._transform_mapping(mapping, pole0, True, len(solution.segments))
            for mapping in solution.mappings
        )
        reflected = canonical.DecoratedSolution(tuple(segments), tuple(points), mappings)
        self.assertEqual(
            canonical.canonicalize_decorated_data(solution).key,
            canonical.canonicalize_decorated_data(reflected).key,
        )

    def test_four_known_voderberg_renamings_share_decorated_key(self):
        targets = {
            215: "(P0 = a0) V0 (a1) V1 (a2 = 0) V1^-1 (P1 = a3) V0 (a1) V1 (a2 = 0) V1^-1 (-a1) V0^-1",
            1423: "(P0 = a0) V0^-1 (a1) V1 (a2 = 0) V1^-1 (-a1) V0 (P1 = a3) V1 (a2 = 0) V1^-1 (-a1) V0",
            1447: "(P0 = a0) V0 (-a1) V1 (a2 = 0) V1^-1 (a1) V0^-1 (P1 = a3) V0 (-a1) V1 (a2 = 0) V1^-1",
            2167: "(P0 = a0) V0 (a1 = 0) V0^-1 (a2) V1 (P1 = a3) V1^-1 (-a2) V0 (a1 = 0) V0^-1 (a2) V1",
        }
        keys = []
        for case_id, target_text in targets.items():
            case = base.find_case(case_id)
            search = formal_equation_audit.explore_case_with_audit(
                case,
                max_depth=5,
                max_states=100,
                max_cycle_unrolls=None,
            )
            matching = []
            for state, _derivation in search.terminal_states:
                angle_solution = analysis_pipeline.solve_point_angles(case, state)
                formal = profile_formatter.build_formal_profile(
                    case, state, angle_solution
                )
                if formal.text == target_text:
                    matching.append(
                        canonical.canonicalize_terminal_solution(
                            case, state, formal
                        ).key
                    )
            self.assertEqual(len(matching), 1, case_id)
            keys.extend(matching)
        self.assertEqual(len(set(keys)), 1)

    def test_relative_copy_parity_remains_part_of_key(self):
        solution = self._solution()
        changed_mapping = replace(
            solution.mappings[0], mirror_sign=base.REFLECTED
        )
        changed = canonical.DecoratedSolution(
            solution.segments,
            solution.points,
            (changed_mapping, solution.mappings[1]),
        )
        self.assertNotEqual(
            canonical.canonicalize_decorated_data(solution).key,
            canonical.canonicalize_decorated_data(changed).key,
        )

    def test_different_segment_pairing_remains_distinct(self):
        solution = self._solution()
        mapping = solution.mappings[1]
        pairs = list(mapping.pairs)
        source, target = pairs[0]
        pairs[0] = (source, canonical.DirectedSegmentRef((target.position + 1) % 6, target.orientation))
        changed_mapping = replace(mapping, pairs=tuple(pairs))
        changed = canonical.DecoratedSolution(
            solution.segments,
            solution.points,
            (solution.mappings[0], changed_mapping),
        )
        self.assertNotEqual(
            canonical.canonicalize_decorated_data(solution).key,
            canonical.canonicalize_decorated_data(changed).key,
        )


if __name__ == "__main__":
    unittest.main()
