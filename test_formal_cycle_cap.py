import unittest

import formal_equation_audit as audit
import symbolic_enumerator as base


class FormalCycleCapTests(unittest.TestCase):
    def test_cap_is_independent_and_reported_as_truncation(self):
        case = base.find_case(59)
        capped = audit.explore_case_with_audit(
            case,
            max_depth=20,
            max_states=10000,
            max_cycle_unrolls=1,
        )
        self.assertTrue(capped.cycle_unroll_cap_enabled)
        self.assertTrue(capped.cycle_unroll_cap_hit)
        self.assertGreater(capped.cycle_unroll_pruned_state_count, 0)
        self.assertTrue(capped.search_truncated)
        data = capped.to_dict()
        self.assertTrue(data["search_truncation_reasons"]["cycle_unroll_cap"])

    def test_zero_style_disable_is_represented_by_none(self):
        case = base.find_case(0)
        uncapped = audit.explore_case_with_audit(
            case,
            max_depth=2,
            max_states=100,
            max_cycle_unrolls=None,
        )
        self.assertFalse(uncapped.cycle_unroll_cap_enabled)
        self.assertFalse(uncapped.cycle_unroll_cap_hit)


if __name__ == "__main__":
    unittest.main()
