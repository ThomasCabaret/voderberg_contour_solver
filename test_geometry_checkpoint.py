import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import geometry_checkpoint as checkpoint


@dataclass(frozen=True)
class _Profile:
    profile_id: int
    case_id: int


class GeometryCheckpointTests(unittest.TestCase):
    def test_transactional_results_resume_across_reopen(self):
        profiles = [_Profile(10, 100), _Profile(11, 101), _Profile(12, 102)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "survivors.json"
            source.write_text('{"profiles":[]}', encoding="utf-8")
            db = root / "checkpoint.sqlite3"
            run_key, identity = checkpoint.build_run_identity(
                input_path=source,
                input_sha256=checkpoint.sha256_file(source),
                selected_profiles=profiles,
                configuration={"intermediate_points": 3, "attempts": 2},
            )
            with checkpoint.GeometryCheckpoint(db) as store:
                summary = store.prepare_run(
                    run_key=run_key,
                    identity=identity,
                    selected_count=len(profiles),
                )
                store.record_result(
                    run_id=summary.run_id,
                    profile_id=10,
                    case_id=100,
                    ordinal=1,
                    status="no_candidate",
                )
                store.record_result(
                    run_id=summary.run_id,
                    profile_id=11,
                    case_id=101,
                    ordinal=2,
                    status="found",
                    candidate={"profile_id": 11, "vertices": [[0, 0], [1, 0]]},
                )

            with checkpoint.GeometryCheckpoint(db) as store:
                resumed = store.prepare_run(
                    run_key=run_key,
                    identity=identity,
                    selected_count=len(profiles),
                )
                self.assertEqual(resumed.completed_count, 2)
                self.assertEqual(resumed.remaining_count, 1)
                self.assertEqual(
                    store.completed_profile_ids(resumed.run_id, retry_errors=False),
                    {10, 11},
                )
                self.assertEqual(store.load_candidates(resumed.run_id)[0]["profile_id"], 11)

    def test_fresh_clears_only_matching_run(self):
        profiles = [_Profile(1, 2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "survivors.json"
            source.write_text("x", encoding="utf-8")
            run_key, identity = checkpoint.build_run_identity(
                input_path=source,
                input_sha256=checkpoint.sha256_file(source),
                selected_profiles=profiles,
                configuration={"intermediate_points": 0},
            )
            with checkpoint.GeometryCheckpoint(root / "resume.sqlite3") as store:
                first = store.prepare_run(
                    run_key=run_key, identity=identity, selected_count=1
                )
                store.record_result(
                    run_id=first.run_id,
                    profile_id=1,
                    case_id=2,
                    ordinal=1,
                    status="error",
                    error_text="test",
                )
                reset = store.prepare_run(
                    run_key=run_key,
                    identity=identity,
                    selected_count=1,
                    fresh=True,
                )
                self.assertEqual(reset.completed_count, 0)


if __name__ == "__main__":
    unittest.main()
