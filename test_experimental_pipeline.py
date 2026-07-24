import unittest

import experimental_pipeline
import joint_translation_z3 as joint


class ExperimentalPipelineTests(unittest.TestCase):
    def test_voderberg_reaches_z3_problem_without_exact_rejection(self):
        case, state = joint.find_voderberg_case_and_state()
        result = experimental_pipeline.analyze_experimental_profile(
            case,
            state,
            prepare_z3=True,
            run_z3=False,
        )
        self.assertEqual(result.status, "z3_problem_ready")
        self.assertFalse(result.exact_encoded_model_rejection)
        self.assertIsNotNone(result.z3_problem)
        self.assertIsNone(result.z3_result)

    def test_z3_execution_is_optional(self):
        case, state = joint.find_voderberg_case_and_state()
        result = experimental_pipeline.analyze_experimental_profile(
            case,
            state,
            prepare_z3=False,
            run_z3=False,
        )
        self.assertEqual(result.status, "external_system_built")
        self.assertIsNone(result.z3_problem)


if __name__ == "__main__":
    unittest.main()
