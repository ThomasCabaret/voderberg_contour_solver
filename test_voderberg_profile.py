import unittest

import symbolic_enumerator as base
from parametric_graph import build_graph


class VoderbergProfileTests(unittest.TestCase):
    @staticmethod
    def find_voderberg_case() -> base.PlacementCase:
        matches = []
        expected_loci = {
            "A_start": "P1",
            "A_end": "P0",
            "B_start": "P1",
            "B_end": "A",
        }

        for case in base.enumerate_placement_cases():
            if (
                case.marker_locus_map() == expected_loci
                and case.a_interior_blocks == (("B_end",),)
                and case.b_interior_blocks == ()
                and case.a_direction == base.REVERSE
                and case.b_direction == base.REVERSE
            ):
                matches.append(case)

        if len(matches) != 1:
            raise AssertionError(
                f"Expected one Voderberg placement, found {len(matches)}"
            )
        return matches[0]

    def test_voderberg_placement_is_enumerated_exactly_once(self):
        case = self.find_voderberg_case()

        self.assertEqual(base.word_to_text(case.a_word), "A0 A1")
        self.assertEqual(base.word_to_text(case.a_target), "A1^-1 A0^-1")
        self.assertEqual(base.word_to_text(case.b_word), "B0")
        self.assertEqual(base.word_to_text(case.b_target), "A1^-1")
        self.assertEqual(
            [equation.to_text() for equation in case.equations],
            [
                "A0 A1 = A1^-1 A0^-1",
                "B0 = A1^-1",
            ],
        )

    def test_every_terminal_voderberg_state_satisfies_both_projections(self):
        case = self.find_voderberg_case()
        terminal_states = list(base.enumerate_terminal_states(case))

        self.assertEqual(len(terminal_states), 3)
        for state, _derivation in terminal_states:
            self.assertTrue(
                base.terminal_state_satisfies_case(case, state),
                msg=f"Invalid terminal environment: {state.environment_map()}",
            )

    def test_exact_two_parameter_voderberg_profile_is_found(self):
        case = self.find_voderberg_case()
        schemes = list(base.enumerate_solution_schemes(case))

        expected_a = (
            base.Literal("X0"),
            base.Literal("X1"),
            base.Literal("X1", True),
            base.Literal("X0", True),
        )
        expected_b = (
            base.Literal("X0"),
            base.Literal("X1"),
            base.Literal("X1", True),
        )
        matching = [
            scheme
            for scheme in schemes
            if scheme.a_expression == expected_a and scheme.b_expression == expected_b
        ]

        self.assertEqual(len(matching), 1)
        self.assertEqual(
            matching[0].derivation,
            (
                "equal_length",
                "left_strictly_shorter",
                "involutive_palindrome",
            ),
        )

    def test_voderberg_derivation_graph_is_finite_and_terminal(self):
        case = self.find_voderberg_case()
        graph = build_graph(
            case.equations,
            initial_a=case.a_word,
            initial_b=case.b_word,
            case_id=case.case_id,
            max_nodes=100,
            max_edges=300,
        )

        self.assertTrue(graph.complete)
        self.assertTrue(any(node.terminal for node in graph.nodes))
        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 5)


if __name__ == "__main__":
    unittest.main()
