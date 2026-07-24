import unittest

import symbolic_enumerator as base
from parametric_graph import (
    build_graph,
    commuting_demo_graph,
    edge_ids_in_cycles,
)


class ParametricGraphTests(unittest.TestCase):
    def test_commuting_equation_has_finite_cyclic_graph(self):
        graph = commuting_demo_graph()
        self.assertTrue(graph.complete)
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 3)
        self.assertEqual(len(edge_ids_in_cycles(graph.nodes, graph.edges)), 2)

    def test_commuting_terminal_edge_identifies_common_root(self):
        graph = commuting_demo_graph()
        terminal_edges = [
            edge
            for edge in graph.edges
            if graph.nodes[edge.target].terminal
        ]
        self.assertEqual(len(terminal_edges), 1)
        morphism = terminal_edges[0].morphism_map()
        self.assertEqual(base.word_to_text(morphism["V0"]), "K0")
        self.assertEqual(base.word_to_text(morphism["V1"]), "K0")

    def test_case_34_graph_is_finite(self):
        case = base.find_case(34)
        graph = build_graph(
            case.equations,
            initial_a=case.a_word,
            initial_b=case.b_word,
            case_id=case.case_id,
            max_nodes=500,
            max_edges=2000,
        )
        self.assertTrue(graph.complete)
        self.assertGreater(len(graph.nodes), 1)
        self.assertTrue(any(node.terminal for node in graph.nodes))

    def test_prefix_order_branches_are_present(self):
        x = base.Literal("X")
        u = base.Literal("U")
        y = base.Literal("Y")
        v = base.Literal("V")
        graph = build_graph(
            equations=(base.Equation((x, u), (y, v)),),
            initial_a=(x, u),
            initial_b=(y, v),
            max_nodes=200,
            max_edges=500,
        )
        branches = {edge.branch for edge in graph.edges}
        self.assertIn("same_cut", branches)
        self.assertIn("left_cut_before_right_cut", branches)
        self.assertIn("right_cut_before_left_cut", branches)


if __name__ == "__main__":
    unittest.main()
