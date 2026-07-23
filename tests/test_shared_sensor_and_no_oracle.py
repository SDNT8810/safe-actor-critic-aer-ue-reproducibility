from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_ac_repro.simulation import (
    CircleObstacle,
    LinearCritic,
    MethodState,
    OnlineCenterEstimator,
    WaypointActor,
    World,
    cbf_filter,
    generate_shared_sensor_stream,
    method_specs,
    simulate_episode,
)


class SharedSensorAndNoOracleTests(unittest.TestCase):
    def test_learned_critic_changes_actor_update(self) -> None:
        world = World()
        actor_a = WaypointActor(world, seed=3)
        actor_b = copy.deepcopy(actor_a)
        critic_a = LinearCritic()
        critic_b = LinearCritic()
        critic_b.w[0] = 100.0
        sample = {
            "x": 1.00, "y": 1.30, "x_next": 1.00, "y_next": 1.30,
            "u_exec_x": 0.0, "u_exec_y": 0.0, "instant_cost": 1.0,
            "exploration_x": WaypointActor.EXPLORATION_STD,
            "exploration_y": 0.0, "actor_waypoint_index": 5,
            "belief_grad_x": 0.0, "belief_grad_y": 0.0,
            "safety_score": 0.0, "uncertainty_score": 0.0,
            "collision_observed": 0,
        }
        actor_a.update_from_samples(world, [sample], critic_a, lr=0.004)
        actor_b.update_from_samples(world, [sample], critic_b, lr=0.004)
        self.assertFalse(np.array_equal(actor_a.waypoints, actor_b.waypoints))

    def test_sensor_stream_is_method_independent(self) -> None:
        world = World()
        a = generate_shared_sensor_stream(
            world, seed=4, episode_id=1001, stress=4.0, n_steps=world.eval_horizon
        )
        b = generate_shared_sensor_stream(
            world, seed=4, episode_id=1001, stress=4.0, n_steps=world.eval_horizon
        )
        np.testing.assert_array_equal(a.raw, b.raw)
        np.testing.assert_array_equal(a.valid, b.valid)
        np.testing.assert_array_equal(a.observable_jump, b.observable_jump)
        self.assertEqual(a.stream_hash, b.stream_hash)

    def test_severe_profile_has_larger_hidden_center_error_than_moderate(self) -> None:
        world = World()
        moderate = generate_shared_sensor_stream(
            world, seed=2, episode_id=1001, stress=2.2, n_steps=world.eval_horizon
        )
        severe = generate_shared_sensor_stream(
            world, seed=2, episode_id=1001, stress=4.0, n_steps=world.eval_horizon
        )
        truth = np.stack([o.c for o in world.uncertain_obstacles])[None, :, :]
        moderate_rmse = float(np.sqrt(np.mean((moderate.raw - truth) ** 2)))
        severe_rmse = float(np.sqrt(np.mean((severe.raw - truth) ** 2)))
        self.assertGreater(severe_rmse, moderate_rmse)

    def test_controller_is_invariant_to_hidden_true_center(self) -> None:
        state = np.array([0.96, 1.68])
        beliefs = [np.array([0.76, 1.52]), np.array([1.89, 1.02])]
        sigmas = [0.024, 0.048]
        for spec in method_specs():
            wa, wb = World(), World()
            wb.uncertain_obstacles = [
                CircleObstacle(o.name, (o.center[0] + 0.43, o.center[1] - 0.31), o.radius)
                for o in wb.uncertain_obstacles
            ]
            aa, ab = WaypointActor(wa, seed=17), WaypointActor(wb, seed=17)
            ua = aa.act(wa, state, beliefs, sigmas, np.zeros(2), train=False)
            ub = ab.act(wb, state, beliefs, sigmas, np.zeros(2), train=False)
            np.testing.assert_allclose(ua, ub, atol=0.0, rtol=0.0)
            ea, *_ = cbf_filter(wa, state, ua, beliefs, sigmas, spec.use_cbf, 0.012)
            eb, *_ = cbf_filter(wb, state, ub, beliefs, sigmas, spec.use_cbf, 0.012)
            np.testing.assert_allclose(ea, eb, atol=0.0, rtol=0.0)

    def test_replay_contains_no_evaluator_truth(self) -> None:
        world = World()
        full = [s for s in method_specs() if s.key == "Full"][0]
        state = MethodState(full, world, seed=0)
        simulate_episode(
            state, episode_id=0, stress=1.0,
            disturbance_stress=1.0, train=True, record=False,
        )
        self.assertGreater(len(state.replay.data), 0)
        forbidden = (
            "true", "hidden_event", "raw_error", "belief_error",
            "disturbance_x", "disturbance_y",
        )
        for tr in state.replay.data:
            bad = [key for key in tr if any(token in key for token in forbidden)]
            self.assertEqual([], bad, msg=f"evaluator-only fields leaked into replay: {bad}")

    def test_full_anchor_does_not_use_true_center_labels(self) -> None:
        world = World()
        full = [s for s in method_specs() if s.key == "Full"][0]
        est_a = OnlineCenterEstimator(world, full)
        shifted = copy.deepcopy(world)
        shifted.uncertain_obstacles = [
            CircleObstacle(o.name, (o.center[0] + 0.5, o.center[1] - 0.4), o.radius)
            for o in shifted.uncertain_obstacles
        ]
        est_b = OnlineCenterEstimator(shifted, full)
        samples = []
        rng = np.random.default_rng(11)
        for _ in range(30):
            tr = {}
            for i, c in enumerate(([0.81, 1.59], [1.83, 0.93]), start=1):
                z = np.asarray(c) + rng.normal(scale=0.01, size=2)
                tr[f"obs{i}_raw_x"], tr[f"obs{i}_raw_y"] = z
            samples.append(tr)
        est_a.replay_calibration(samples)
        est_b.replay_calibration(samples)
        for ca, cb in zip(est_a.learned_anchor, est_b.learned_anchor):
            np.testing.assert_allclose(ca, cb, atol=0.0, rtol=0.0)

    def test_known_map_discrete_guard_prevents_wall_crossing(self) -> None:
        world = World()
        p = np.array([0.35, 1.50])  # 1.8 cm from inflated left face of mapped column
        u_nom = np.array([world.max_speed, 0.0])
        beliefs = [np.array([-2.0, -2.0]), np.array([5.0, 5.0])]
        sigmas = [0.01, 0.01]
        u, *_ = cbf_filter(
            world, p, u_nom, beliefs, sigmas, use_cbf=True, disturbance_bound=0.012
        )
        q = p + world.dt * u
        h, _, _ = world.known_static_clearance(q)
        self.assertGreaterEqual(h, 0.0)


if __name__ == "__main__":
    unittest.main()
