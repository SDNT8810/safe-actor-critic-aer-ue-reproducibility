from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from safe_ac_static_obstacle_uncertainty_v17 import World


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "demo_results"


class ReleaseContractTests(unittest.TestCase):
    def test_registered_evaluation_horizon_is_unchanged(self) -> None:
        self.assertEqual(World.eval_horizon, 280)
        self.assertAlmostEqual(World.eval_horizon * World.dt, 15.4, places=12)

    def test_display_trajectory_contains_full_horizon(self) -> None:
        trajectories = pd.read_csv(
            RESULTS / "evaluation_trajectories_display_seed4.csv"
        )
        stress = trajectories[trajectories.eval_type == "stress"]
        self.assertEqual(set(stress.method.unique()), {
            "AC", "AC+CBF", "AC+CBF+PER", "AC+CBF+UE", "AC+CBF+AER", "Full"
        })
        for _, method_rows in stress.groupby("method"):
            self.assertEqual(len(method_rows), 280)
            self.assertAlmostEqual(float(method_rows.time.max()), 15.345, places=12)

            distance = np.hypot(
                method_rows.x.to_numpy() - World().goal[0],
                method_rows.y.to_numpy() - World().goal[1],
            )
            reached_times = method_rows.time.to_numpy()[distance < World.goal_tolerance]
            self.assertGreater(len(reached_times), 0)
            self.assertLessEqual(float(reached_times[0]), 10.0)

    def test_bundled_headline_result_matches_manuscript(self) -> None:
        metrics = pd.read_csv(RESULTS / "stress_eval_metrics.csv")
        full = metrics[metrics.method == "Full"]
        self.assertEqual(int(full.violation_count.sum()), 0)
        self.assertEqual(int(full.reached.sum()), 5)
        self.assertAlmostEqual(float(full.cost.mean()), 7.632, places=3)
        self.assertAlmostEqual(
            float(full.belief_center_rmse_cm.mean()), 3.517, places=3
        )


if __name__ == "__main__":
    unittest.main()
