import unittest

import formal_equation_audit as audit
import symbolic_enumerator as symbolic


class FormalEquationAuditTests(unittest.TestCase):
    def test_structure_counts_match_equation_literals(self):
        case = next(symbolic.enumerate_placement_cases())
        result = audit.analyze_case_structure(case)
        expected = sum(
            len(equation.left) + len(equation.right)
            for equation in case.equations
        )
        self.assertEqual(result["total_literal_occurrences"], expected)
        self.assertEqual(
            max(result["occurrence_count_by_variable"].values()),
            result["maximum_variable_occurrence_count"],
        )

    def test_bounded_explorer_matches_existing_terminal_enumerator(self):
        case = symbolic.find_case(34)
        expected = list(
            symbolic.enumerate_terminal_states(case, max_depth=5, max_states=100)
        )
        observed = audit.explore_case_with_audit(
            case, max_depth=5, max_states=100
        )
        self.assertEqual(len(observed.terminal_states), len(expected))
        self.assertEqual(
            [(state.equations, derivation) for state, derivation in observed.terminal_states],
            [(state.equations, derivation) for state, derivation in expected],
        )

    def test_summary_reports_quadratic_partition(self):
        records = []
        for case in list(symbolic.enumerate_placement_cases())[:5]:
            search = audit.explore_case_with_audit(
                case, max_depth=1, max_states=10
            )
            records.append(
                {
                    "structure": audit.analyze_case_structure(case),
                    "bounded_search": search.to_dict(),
                }
            )
        summary = audit.summarize_case_audits(records)
        self.assertEqual(summary["placement_system_count"], 5)
        self.assertEqual(
            summary["quadratic_system_count"]
            + summary["nonquadratic_system_count"],
            5,
        )

    def test_search_outcome_flags_distinguish_partial_and_unresolved(self):
        truncated_with_terminal = audit.SolverSearchAudit(
            terminal_states=((None, ()),),
            visited_states=10,
            unique_depth_states=10,
            duplicate_depth_states_skipped=0,
            maximum_depth_reached=5,
            depth_frontier_cut_count=1,
            state_limit_hit=False,
            queue_size_when_state_limit_hit=0,
            residual_equation_revisit_count=0,
            structural_state_revisit_count=0,
            terminal_count=1,
            branch_counts=(),
        )
        self.assertEqual(
            truncated_with_terminal.search_outcome_class,
            "truncated_with_terminal_profiles",
        )
        self.assertTrue(truncated_with_terminal.terminal_profile_set_may_be_incomplete)
        self.assertFalse(truncated_with_terminal.existence_unresolved_by_bounds)

        truncated_without_terminal = audit.SolverSearchAudit(
            terminal_states=(),
            visited_states=10,
            unique_depth_states=10,
            duplicate_depth_states_skipped=0,
            maximum_depth_reached=5,
            depth_frontier_cut_count=0,
            state_limit_hit=True,
            queue_size_when_state_limit_hit=3,
            residual_equation_revisit_count=0,
            structural_state_revisit_count=0,
            terminal_count=0,
            branch_counts=(),
        )
        self.assertEqual(
            truncated_without_terminal.search_outcome_class,
            "truncated_without_terminal_profiles",
        )
        self.assertTrue(truncated_without_terminal.existence_unresolved_by_bounds)

    def test_summary_contains_full_terminal_truncation_cross_table(self):
        outcomes = [
            ("initially_inconsistent", 0, True, False),
            ("exhausted_with_terminal_profiles", 1, False, False),
            ("truncated_with_terminal_profiles", 2, False, True),
            ("exhausted_without_terminal_profiles", 0, False, False),
            ("truncated_without_terminal_profiles", 0, False, True),
        ]
        records = []
        for outcome, terminal_count, initial, truncated in outcomes:
            records.append({
                "structure": {
                    "occurrence_class": "quadratic",
                    "variable_count": 2,
                    "maximum_variable_occurrence_count": 2,
                    "recommended_complete_solver_family": "quadratic_nielsen_graph_candidate",
                    "is_quadratic_system": True,
                    "contains_involution": True,
                    "at_most_once_per_equation_side": True,
                },
                "bounded_search": {
                    "search_outcome_class": outcome,
                    "terminal_count": terminal_count,
                    "initial_inconsistent": initial,
                    "search_truncated": truncated,
                    "state_limit_hit": truncated,
                    "depth_frontier_cut_count": 0,
                    "search_exhausted_within_current_graph": (not initial and not truncated),
                },
            })
        summary = audit.summarize_case_audits(records)
        self.assertEqual(summary["truncated_with_terminal_profiles_case_count"], 1)
        self.assertEqual(summary["truncated_without_terminal_profiles_case_count"], 1)
        self.assertEqual(summary["exhausted_with_terminal_profiles_case_count"], 1)
        self.assertEqual(summary["exhausted_without_terminal_profiles_case_count"], 1)
        self.assertEqual(summary["initially_inconsistent_case_count"], 1)
        self.assertEqual(summary["submitted_to_branching_solver_case_count"], 4)


if __name__ == "__main__":
    unittest.main()
