#!/usr/bin/env python3
"""V17.1 severe shared-sensor benchmark for integrated safe actor--critic control.

Information-flow contract
-------------------------
1. The two uncertain circular obstacles are physically static.  Their true centers
   are hidden from every controller and used only to generate sensor data and to
   evaluate true collision/clearance.
2. For a fixed seed and episode, every method receives exactly the same raw
   obstacle-center measurements, validity bits, exploration sequence, and plant
   disturbance sequence.
3. Methods without UE use the raw/held center directly.  UE filters the same stream.
   Full uses the same filter plus a robust static-center anchor learned from raw
   measurements sampled by adaptive replay.  No true-center label is used.
4. The finite replay buffer stores only controller-visible quantities.  Evaluator
   truth is kept only in recorded trajectory rows.
5. The value critic learns from the realized cost and next state produced by the
   executed action.  Its TD residual drives a replay-weighted score-function
   waypoint update based on the common Gaussian exploration draw.  Because
   replayed samples have no behavior-policy likelihood ratio or importance
   correction, this is a biased off-policy surrogate, not an unbiased on-policy
   gradient.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

try:
    import scipy
    from scipy.stats import friedmanchisquare, wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    n = norm(v)
    if n > 1e-12:
        return v / n
    return np.array([1.0, 0.0]) if fallback is None else fallback.copy()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def geometric_median(points: np.ndarray, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Weiszfeld geometric median used for robust replay calibration."""
    if len(points) == 0:
        raise ValueError("geometric_median requires at least one point")
    x = np.median(points, axis=0).astype(float)
    for _ in range(max_iter):
        d = np.linalg.norm(points - x, axis=1)
        if np.any(d < 1e-12):
            return points[int(np.argmin(d))].copy()
        w = 1.0 / np.maximum(d, 1e-12)
        x_new = np.sum(points * w[:, None], axis=0) / np.sum(w)
        if norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


@dataclass(frozen=True)
class RectObstacle:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0])

    @property
    def half(self) -> np.ndarray:
        return np.array([(self.xmax - self.xmin) / 2.0, (self.ymax - self.ymin) / 2.0])

    def signed_distance_and_grad(self, p: np.ndarray, inflation: float = 0.0) -> Tuple[float, np.ndarray]:
        c = self.center
        half = self.half + inflation
        q = np.abs(p - c) - half
        outside = np.maximum(q, 0.0)
        d = norm(outside)
        signs = np.sign(p - c)
        signs[signs == 0.0] = 1.0
        if d > 1e-12:
            return d, outside * signs / d
        d_to_side = half - np.abs(p - c)
        axis = int(np.argmin(d_to_side))
        g = np.zeros(2)
        g[axis] = signs[axis]
        return -float(d_to_side[axis]), g


@dataclass(frozen=True)
class CircleObstacle:
    name: str
    center: Tuple[float, float]
    radius: float

    @property
    def c(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float)

    def signed_distance_and_grad(self, p: np.ndarray, inflation: float = 0.0) -> Tuple[float, np.ndarray]:
        diff = p - self.c
        d = norm(diff)
        return d - self.radius - inflation, unit(diff)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    short: str
    label: str
    color: str
    use_cbf: bool
    use_per: bool
    use_ue: bool
    use_aer: bool
    line_width: float

    @property
    def replay_mode(self) -> str:
        if self.use_aer:
            return "aer"
        if self.use_per:
            return "per"
        return "uniform"


METHOD_ORDER = ["AC", "AC+CBF", "AC+CBF+PER", "AC+CBF+UE", "AC+CBF+AER", "Full"]


def method_specs() -> List[MethodSpec]:
    return [
        MethodSpec("AC", "AC", "actor-critic + uniform replay", "#d95f02", False, False, False, False, 2.2),
        MethodSpec("AC+CBF", "AC+CBF", "actor-critic + CBF + uniform replay", "#1b9e77", True, False, False, False, 2.3),
        MethodSpec("AC+CBF+PER", "AC+CBF+PER", "actor-critic + CBF + TD-priority replay", "#7570b3", True, True, False, False, 2.3),
        MethodSpec("AC+CBF+UE", "AC+CBF+UE", "actor-critic + CBF + online uncertainty estimation", "#66a61e", True, False, True, False, 2.4),
        MethodSpec("AC+CBF+AER", "AC+CBF+AER", "actor-critic + CBF + adaptive replay", "#e7298a", True, False, False, True, 2.7),
        MethodSpec("Full", "Full", "actor-critic + CBF + online uncertainty estimation + adaptive replay", "#1f78b4", True, False, True, True, 3.7),
    ]


class World:
    width = 3.20
    height = 3.20
    robot_radius = 0.062
    dt = 0.055
    train_horizon = 115
    eval_horizon = 280
    max_speed = 0.72
    goal_tolerance = 0.055

    def __init__(self) -> None:
        self.start = np.array([0.22, 2.82], dtype=float)
        self.goal = np.array([1.62, 1.58], dtype=float)
        self.rectangles: List[RectObstacle] = [
            RectObstacle("top_left_u_bottom", 0.43, 2.50, 1.20, 2.64),
            RectObstacle("top_left_u_left", 0.43, 2.50, 0.57, 2.88),
            RectObstacle("top_left_u_right", 1.06, 2.50, 1.20, 2.88),
            RectObstacle("top_right_u_bottom", 2.05, 2.48, 2.88, 2.62),
            RectObstacle("top_right_u_left", 2.05, 2.48, 2.19, 2.86),
            RectObstacle("top_right_u_right", 2.74, 2.48, 2.88, 2.86),
            RectObstacle("left_mid_column", 0.43, 1.22, 0.70, 1.80),
            RectObstacle("central_left_wall", 1.18, 1.02, 1.34, 2.08),
            RectObstacle("central_top_wall", 1.18, 1.94, 2.02, 2.08),
            RectObstacle("central_bottom_wall", 1.18, 1.02, 1.68, 1.18),
            RectObstacle("central_right_lip", 1.88, 1.48, 2.02, 2.08),
            RectObstacle("bottom_bar", 0.53, 0.45, 1.72, 0.60),
            RectObstacle("right_column", 2.45, 0.65, 2.75, 1.45),
        ]
        # Evaluator-only physical truth. These obstacles never move.
        self.uncertain_obstacles: List[CircleObstacle] = [
            CircleObstacle("uncertain_worker_mid", (0.82, 1.60), 0.070),
            CircleObstacle("uncertain_cart_lower", (1.82, 0.92), 0.100),
        ]
        # Common map-feasible corridor. It does not encode exact uncertain-obstacle avoidance.
        self.initial_waypoints = np.array([
            [0.22, 2.82], [0.31, 2.43], [0.62, 2.25], [0.87, 1.99],
            [0.90, 1.62], [1.04, 1.27], [1.06, 0.82], [1.44, 0.70],
            [2.18, 0.70], [2.34, 1.14], [1.82, 1.39], [1.62, 1.58],
        ], dtype=float)

    def known_static_clearance(self, p: np.ndarray) -> Tuple[float, str, np.ndarray]:
        vals: List[Tuple[float, str, np.ndarray]] = []
        for r in self.rectangles:
            h, g = r.signed_distance_and_grad(p, inflation=self.robot_radius)
            vals.append((h, r.name, g))
        vals += [
            (p[0] - self.robot_radius, "world_left", np.array([1.0, 0.0])),
            (self.width - self.robot_radius - p[0], "world_right", np.array([-1.0, 0.0])),
            (p[1] - self.robot_radius, "world_bottom", np.array([0.0, 1.0])),
            (self.height - self.robot_radius - p[1], "world_top", np.array([0.0, -1.0])),
        ]
        return min(vals, key=lambda z: z[0])

    def uncertain_true_clearance(self, p: np.ndarray) -> Tuple[float, str, np.ndarray]:
        vals: List[Tuple[float, str, np.ndarray]] = []
        for obs in self.uncertain_obstacles:
            h, g = obs.signed_distance_and_grad(p, inflation=self.robot_radius)
            vals.append((h, obs.name, g))
        return min(vals, key=lambda z: z[0])

    def true_clearance(self, p: np.ndarray) -> Tuple[float, str, np.ndarray]:
        return min([self.known_static_clearance(p), self.uncertain_true_clearance(p)], key=lambda z: z[0])


@dataclass(frozen=True)
class SensorStream:
    raw: np.ndarray                 # [time, obstacle, xy]
    valid: np.ndarray               # [time, obstacle]
    observable_jump: np.ndarray     # detector-visible frame-to-frame change
    hidden_event: np.ndarray        # evaluator diagnostic only
    reported_sigma: np.ndarray      # detector-reported standard deviation
    stream_hash: str


def _episode_rng(seed: int, episode_id: int) -> np.random.Generator:
    return np.random.default_rng(17_000_003 + 100_003 * seed + 1009 * episode_id)


def generate_shared_sensor_stream(
    world: World,
    seed: int,
    episode_id: int,
    stress: float,
    n_steps: int,
) -> SensorStream:
    """Generate one method-independent raw center stream.

    True centers remain fixed.  ``stress`` scales only the sensing mismatch:
    episode registration bias, colored jitter, zero-mean drift, short holds, and
    overconfident piecewise jumps.  All methods get the same arrays.
    """
    rng = _episode_rng(seed, episode_id)
    n_obs = len(world.uncertain_obstacles)
    raw = np.zeros((n_steps, n_obs, 2), dtype=float)
    valid = np.ones((n_steps, n_obs), dtype=int)
    observable_jump = np.zeros((n_steps, n_obs), dtype=float)
    hidden_event = np.zeros((n_steps, n_obs), dtype=int)
    reported_sigma = np.full((n_steps, n_obs), 0.010, dtype=float)

    colored = rng.normal(scale=0.002, size=(n_obs, 2))
    jump_state = np.zeros((n_obs, 2), dtype=float)
    jump_left = np.zeros(n_obs, dtype=int)
    hold_left = np.zeros(n_obs, dtype=int)
    prev = np.stack([obs.c for obs in world.uncertain_obstacles], axis=0)
    episode_bias = np.zeros((n_obs, 2), dtype=float)
    for i in range(n_obs):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        amp = stress * rng.uniform(0.010, 0.019)
        episode_bias[i] = amp * np.array([math.cos(angle), math.sin(angle)])

    base_events = [[54, 72, 188], [136, 151, 205]]
    adverse_directions = [unit(np.array([-1.0, -0.80])), unit(np.array([-0.15, 1.0]))]
    event_starts: Dict[int, List[int]] = {}
    for i in range(n_obs):
        times: List[int] = []
        for b in base_events[i]:
            if b < n_steps:
                times.append(int(np.clip(b + rng.integers(-5, 6), 10, n_steps - 3)))
        if n_steps <= world.train_horizon:
            times.append(int(np.clip(n_steps * (0.50 + 0.08 * i) + rng.integers(-4, 5), 12, n_steps - 4)))
        event_starts[i] = sorted(set(times))

    for k in range(n_steps):
        for i, obs in enumerate(world.uncertain_obstacles):
            phase = 0.075 * k + 0.41 * seed + 0.73 * i + 0.11 * episode_id
            colored[i] = 0.82 * colored[i] + rng.normal(
                scale=0.0032 * math.sqrt(max(stress, 0.05)), size=2
            )
            drift = stress * np.array([
                0.0090 * math.sin(phase),
                0.0080 * math.cos(0.83 * phase + 0.6 * i),
            ])
            if k in event_starts[i]:
                base_angle = math.atan2(adverse_directions[i][1], adverse_directions[i][0])
                angle = base_angle + rng.normal(scale=0.22)
                amp = stress * rng.uniform(0.050, 0.070)
                jump_state[i] = amp * np.array([math.cos(angle), math.sin(angle)])
                jump_left[i] = int(rng.integers(15, 22))
                hidden_event[k, i] = 1
            elif jump_left[i] > 0:
                jump_state[i] *= 0.955
                jump_left[i] -= 1
                hidden_event[k, i] = 1
            else:
                jump_state[i] *= 0.35

            if hold_left[i] <= 0 and k > 18 and rng.random() < min(0.012 * stress, 0.055):
                hold_left[i] = int(rng.integers(2, 5))
            if hold_left[i] > 0:
                candidate = prev[i].copy()
                valid[k, i] = 0
                hold_left[i] -= 1
            else:
                white = rng.normal(scale=0.0026 * math.sqrt(max(stress, 0.05)), size=2)
                candidate = obs.c + episode_bias[i] + drift + colored[i] + jump_state[i] + white
            raw[k, i] = candidate
            observable_jump[k, i] = min(1.0, norm(candidate - prev[i]) / 0.050)
            prev[i] = candidate

    # One run-local digest covers every controller-visible sensor input.  Tiny
    # cross-platform floating-point differences can change it, so it certifies
    # within-run equality across methods rather than a universal byte identity.
    digest_payload = b"".join([
        np.ascontiguousarray(raw).tobytes(),
        np.ascontiguousarray(valid).tobytes(),
        np.ascontiguousarray(observable_jump).tobytes(),
        np.ascontiguousarray(reported_sigma).tobytes(),
    ])
    digest = hashlib.sha256(digest_payload).hexdigest()
    return SensorStream(raw, valid, observable_jump, hidden_event, reported_sigma, digest)


def generate_shared_disturbance_stream(seed: int, episode_id: int, stress: float, n_steps: int) -> np.ndarray:
    """Small exogenous velocity disturbance, separate from sensing stress."""
    rng = np.random.default_rng(19_001_007 + 200_003 * seed + 1013 * episode_id)
    d = np.zeros((n_steps, 2), dtype=float)
    state = np.zeros(2)
    for k in range(n_steps):
        state = 0.90 * state + rng.normal(scale=0.0018 * stress, size=2)
        periodic = stress * np.array([
            0.0040 * math.sin(0.09 * k + 0.2 * seed),
            -0.0035 * math.cos(0.07 * k - 0.1 * seed),
        ])
        d[k] = state + periodic
    return d


def known_map_avoidance_velocity(world: World, p: np.ndarray) -> np.ndarray:
    push = np.zeros(2)
    influence = 0.18
    for r in world.rectangles:
        h, g = r.signed_distance_and_grad(p, inflation=world.robot_radius + 0.022)
        if h < influence:
            push += 0.42 * ((influence - h) / influence) ** 2 * g
    margin = world.robot_radius + 0.025
    walls = [
        (p[0] - margin, np.array([1.0, 0.0])),
        (world.width - margin - p[0], np.array([-1.0, 0.0])),
        (p[1] - margin, np.array([0.0, 1.0])),
        (world.height - margin - p[1], np.array([0.0, -1.0])),
    ]
    for h, g in walls:
        if h < 0.14:
            push += 0.34 * ((0.14 - h) / 0.14) ** 2 * g
    return push


def perceived_obstacle_avoidance_velocity(
    world: World,
    p: np.ndarray,
    centers: Sequence[np.ndarray],
    sigmas: Sequence[float],
) -> np.ndarray:
    push = np.zeros(2)
    influence = 0.31
    for obs, center, sigma in zip(world.uncertain_obstacles, centers, sigmas):
        diff = p - center
        h = norm(diff) - (obs.radius + world.robot_radius + 0.010 + 0.35 * sigma)
        if h < influence:
            push += 0.31 * ((influence - h) / influence) ** 2 * unit(diff)
    return push


class WaypointActor:
    EXPLORATION_STD = 0.017
    TD_POLICY_GAIN = 0.18

    def __init__(self, world: World, seed: int) -> None:
        rng = np.random.default_rng(12_345 + seed)
        self.waypoints = world.initial_waypoints.copy()
        jitter = rng.normal(scale=0.006, size=self.waypoints.shape)
        jitter[0] = 0.0
        jitter[-1] = 0.0
        self.waypoints += jitter
        self.wp_idx = 1

    def reset_episode(self) -> None:
        self.wp_idx = 1

    def act(
        self,
        world: World,
        p: np.ndarray,
        belief_centers: Sequence[np.ndarray],
        belief_sigmas: Sequence[float],
        exploration: np.ndarray,
        train: bool,
    ) -> np.ndarray:
        while self.wp_idx < len(self.waypoints) - 1:
            d_cur = norm(self.waypoints[self.wp_idx] - p)
            d_next = norm(self.waypoints[self.wp_idx + 1] - p)
            if d_cur < 0.118 or d_next + 0.030 < d_cur:
                self.wp_idx += 1
            else:
                break
        target = self.waypoints[self.wp_idx]
        u = 2.25 * (target - p)
        if self.wp_idx >= len(self.waypoints) - 2:
            u += 0.28 * (world.goal - p)
        u += known_map_avoidance_velocity(world, p)
        u += perceived_obstacle_avoidance_velocity(world, p, belief_centers, belief_sigmas)
        if train:
            u += exploration
        speed = norm(u)
        return u if speed <= world.max_speed else world.max_speed * u / speed

    def update_from_samples(
        self,
        world: World,
        samples: Sequence[dict],
        critic: "LinearCritic",
        lr: float,
    ) -> None:
        """Update waypoints with a replay-weighted score-function surrogate.

        The Gaussian exploration draw is the score-function direction.  The
        cost TD residual is recomputed by the learned critic from the executed
        (post-CBF) action, so an exploratory displacement with positive cost
        advantage moves the waypoint mean in the opposite direction.  The
        existing controller-visible barrier gradient remains an auxiliary
        safety-shaping term shared by all configurations.  Replayed samples do
        not carry behavior-policy likelihoods and no importance ratios are
        applied, so this clipped update is intentionally a biased off-policy
        surrogate rather than an unbiased policy-gradient estimator.
        """
        if not samples:
            return
        for tr in samples:
            p = np.array([tr["x"], tr["y"]], dtype=float)
            j = int(tr.get(
                "actor_waypoint_index",
                int(np.argmin(np.linalg.norm(self.waypoints[1:-1] - p, axis=1))) + 1,
            ))
            j = int(np.clip(j, 1, len(self.waypoints) - 2))

            p_next = np.array([tr["x_next"], tr["y_next"]], dtype=float)
            u_exec = np.array([tr["u_exec_x"], tr["u_exec_y"]], dtype=float)
            cost_td = critic.td_error(
                world, p, u_exec, float(tr["instant_cost"]), p_next,
            )
            exploration = np.array([
                tr["exploration_x"], tr["exploration_y"],
            ], dtype=float)
            score_direction = np.clip(
                exploration / self.EXPLORATION_STD, -2.5, 2.5,
            )
            policy_step = (
                -self.TD_POLICY_GAIN
                * clamp(cost_td, -2.2, 2.2)
                * score_direction
            )

            grad = np.array([tr["belief_grad_x"], tr["belief_grad_y"]], dtype=float)
            pressure = (
                0.58 * float(tr["safety_score"])
                + 0.27 * float(tr["uncertainty_score"])
                + 0.15 * float(tr["collision_observed"])
            )
            safety_step = np.zeros(2) if norm(grad) < 1e-9 else pressure * unit(grad)
            self.waypoints[j] += lr * (policy_step + safety_step)
        smoothed = self.waypoints.copy()
        for j in range(1, len(self.waypoints) - 1):
            smoothed[j] = 0.74 * self.waypoints[j] + 0.13 * self.waypoints[j - 1] + 0.13 * self.waypoints[j + 1]
        self.waypoints = smoothed
        corridor_bounds = {
            1: (0.24, 0.38, 2.31, 2.46),
            2: (0.53, 0.77, 2.10, 2.36),
            3: (0.77, 1.07, 1.84, 2.14),
            4: (0.84, 1.13, 1.52, 1.86),
            5: (0.88, 1.11, 1.16, 1.47),
            6: (0.97, 1.11, 0.73, 0.89),
            7: (1.29, 1.60, 0.65, 0.77),
            8: (2.02, 2.31, 0.65, 0.78),
            9: (2.16, 2.43, 1.00, 1.26),
            10: (1.70, 1.94, 1.33, 1.55),
        }
        for j, (xmin, xmax, ymin, ymax) in corridor_bounds.items():
            self.waypoints[j, 0] = np.clip(self.waypoints[j, 0], xmin, xmax)
            self.waypoints[j, 1] = np.clip(self.waypoints[j, 1], ymin, ymax)
        self.waypoints[0] = world.start
        self.waypoints[-1] = world.goal


class LinearCritic:
    """Linear state-value critic for the executed closed-loop transition."""

    def __init__(self) -> None:
        self.w = np.zeros(7, dtype=float)
        self.gamma = 0.985
        self.lr = 0.016

    @staticmethod
    def features(world: World, p: np.ndarray, u_exec: Optional[np.ndarray] = None) -> np.ndarray:
        g = world.goal - p
        return np.array([
            1.0, g[0], g[1], g[0] ** 2, g[1] ** 2,
            p[0] / world.width, p[1] / world.height,
        ], dtype=float)

    def value(self, world: World, p: np.ndarray, u_exec: Optional[np.ndarray] = None) -> float:
        return float(self.w @ self.features(world, p, u_exec))

    def td_error(self, world: World, p: np.ndarray, u_exec: np.ndarray, cost: float, p_next: np.ndarray) -> float:
        # The executed action determines both ``cost`` and ``p_next``.  A state-
        # value critic is used so this Bellman residual is a conventional cost
        # advantage for the score-function actor update.
        return float(cost + self.gamma * self.value(world, p_next) - self.value(world, p))

    def update(self, world: World, samples: Sequence[dict]) -> None:
        for tr in samples:
            p = np.array([tr["x"], tr["y"]])
            pn = np.array([tr["x_next"], tr["y_next"]])
            phi = self.features(world, p)
            td = float(tr["instant_cost"]) + self.gamma * self.value(world, pn) - float(self.w @ phi)
            self.w += self.lr * clamp(td, -2.2, 2.2) * phi / (1.0 + float(phi @ phi))


class OnlineCenterEstimator:
    """Robust constant-position filter with optional replay-calibrated anchor."""

    def __init__(self, world: World, spec: MethodSpec) -> None:
        self.world = world
        self.spec = spec
        n = len(world.uncertain_obstacles)
        self.learned_anchor: List[Optional[np.ndarray]] = [None for _ in range(n)]
        self.learned_nominal_scale = [0.018 for _ in range(n)]
        self.learned_gate = [0.050 for _ in range(n)]
        self.anchor_updates = [0 for _ in range(n)]
        self.center = [np.zeros(2) for _ in range(n)]
        self.var = [0.025 ** 2 for _ in range(n)]
        self.prev_raw = [np.zeros(2) for _ in range(n)]
        self.initialized = False

    def reset_episode(self, first_raw: np.ndarray) -> None:
        for i in range(len(self.center)):
            if self.spec.use_ue and self.learned_anchor[i] is not None:
                self.center[i] = 0.78 * self.learned_anchor[i] + 0.22 * first_raw[i]
            else:
                self.center[i] = first_raw[i].copy()
            self.var[i] = max(self.learned_nominal_scale[i], 0.014) ** 2
            self.prev_raw[i] = first_raw[i].copy()
        self.initialized = True

    def observe(
        self,
        raw_centers: np.ndarray,
        valid: np.ndarray,
        reported_sigma: np.ndarray,
        observable_jump: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float], List[float], List[int]]:
        if not self.initialized:
            self.reset_episode(raw_centers)
        if not self.spec.use_ue:
            centers = [raw_centers[i].copy() for i in range(len(raw_centers))]
            sigmas = [float(reported_sigma[i]) for i in range(len(raw_centers))]
            innovations = [norm(raw_centers[i] - self.prev_raw[i]) for i in range(len(raw_centers))]
            outliers = [int(observable_jump[i] > 0.60 or valid[i] == 0) for i in range(len(raw_centers))]
            self.prev_raw = [r.copy() for r in raw_centers]
            return centers, sigmas, innovations, outliers

        centers: List[np.ndarray] = []
        sigmas: List[float] = []
        innovations: List[float] = []
        outliers: List[int] = []
        for i, raw in enumerate(raw_centers):
            pred = self.center[i]
            innovation = raw - pred
            innovation_norm = norm(innovation)
            gate = self.learned_gate[i] if (self.spec.use_aer and self.learned_anchor[i] is not None) else 0.060
            is_outlier = int(innovation_norm > gate or observable_jump[i] > 0.72 or valid[i] == 0)
            robust_weight = min(1.0, gate / max(innovation_norm, 1e-9))
            base_gain = 0.24 if not self.spec.use_aer else 0.18
            gain = base_gain * robust_weight / (1.0 + innovation_norm / max(3.0 * self.learned_nominal_scale[i], 0.035))
            if valid[i] == 0:
                gain = 0.0
            self.center[i] = pred + gain * innovation
            if self.spec.use_aer and self.learned_anchor[i] is not None:
                anchor_pull = 0.018 + 0.10 * is_outlier
                self.center[i] = (1.0 - anchor_pull) * self.center[i] + anchor_pull * self.learned_anchor[i]
            clipped_sq = min(innovation_norm, gate) ** 2
            self.var[i] = 0.93 * self.var[i] + 0.07 * clipped_sq
            sigma = 0.008 + 1.35 * math.sqrt(max(self.var[i], 1e-8))
            if is_outlier:
                sigma += 0.012 if not self.spec.use_aer else 0.008
            sigma = clamp(sigma, 0.014, 0.075)
            centers.append(self.center[i].copy())
            sigmas.append(sigma)
            innovations.append(innovation_norm)
            outliers.append(is_outlier)
            self.prev_raw[i] = raw.copy()
        return centers, sigmas, innovations, outliers

    def replay_calibration(self, samples: Sequence[dict]) -> None:
        """Full-only self-supervised calibration from replayed raw measurements."""
        if not (self.spec.use_ue and self.spec.use_aer) or not samples:
            return
        for i in range(len(self.center)):
            pts = np.array([[tr[f"obs{i+1}_raw_x"], tr[f"obs{i+1}_raw_y"]] for tr in samples], dtype=float)
            if len(pts) < 8:
                continue
            candidate = geometric_median(pts)
            dist = np.linalg.norm(pts - candidate, axis=1)
            lower = dist[dist <= np.quantile(dist, 0.60)]
            nominal = float(np.median(lower)) if len(lower) else float(np.median(dist))
            nominal = clamp(1.45 * nominal, 0.009, 0.030)
            gate = clamp(2.7 * nominal, 0.030, 0.058)
            if self.learned_anchor[i] is None:
                self.learned_anchor[i] = candidate
            else:
                self.learned_anchor[i] = 0.86 * self.learned_anchor[i] + 0.14 * candidate
            self.learned_nominal_scale[i] = 0.88 * self.learned_nominal_scale[i] + 0.12 * nominal
            self.learned_gate[i] = 0.88 * self.learned_gate[i] + 0.12 * gate
            self.anchor_updates[i] += 1


class ReplayBuffer:
    AER_WEIGHTS = (0.18, 0.39, 0.30, 0.13)  # TD, safety, uncertainty, novelty

    def __init__(self, capacity: int, spec: MethodSpec, seed: int) -> None:
        self.capacity = capacity
        self.spec = spec
        # Use a common base RNG for each seed.  Methods still obtain different
        # samples when their priorities/buffer contents differ, but method names
        # no longer introduce an avoidable random-number confound.
        self.rng = np.random.default_rng(202_401 + 101 * seed)
        self.data: List[dict] = []
        self.generated_count = 0
        self.generated_boundary = 0
        self.generated_uncertainty = 0
        self.sampled_count = 0
        self.sampled_boundary = 0
        self.sampled_uncertainty = 0

    def _priority(self, tr: dict) -> float:
        if self.spec.replay_mode == "uniform":
            return 1.0
        if self.spec.replay_mode == "per":
            return float(tr["td_score"] + 1e-5)
        w_td, w_safe, w_unc, w_nov = self.AER_WEIGHTS
        return float(
            w_td * tr["td_score"] + w_safe * tr["safety_score"]
            + w_unc * tr["uncertainty_score"] + w_nov * tr["novelty_score"] + 1e-5
        )

    def add(self, transition: dict) -> None:
        tr = dict(transition)
        tr["priority"] = self._priority(tr)
        self.generated_count += 1
        self.generated_boundary += int(tr["is_boundary"])
        self.generated_uncertainty += int(tr["is_uncertainty"])
        if len(self.data) < self.capacity:
            self.data.append(tr)
            return
        if self.spec.replay_mode == "uniform":
            j = int(self.rng.integers(0, self.generated_count))
            if j < self.capacity:
                self.data[j] = tr
            return
        j = int(self.rng.integers(0, self.capacity)) if self.rng.random() < 0.16 else int(np.argmin([d["priority"] for d in self.data]))
        if tr["priority"] >= self.data[j]["priority"] or self.rng.random() < 0.035:
            self.data[j] = tr

    def sample(self, batch_size: int) -> List[dict]:
        if not self.data:
            return []
        n = min(batch_size, len(self.data))
        if self.spec.replay_mode == "uniform":
            idx = self.rng.choice(len(self.data), size=n, replace=False)
        else:
            pri = np.array([max(d["priority"], 1e-6) for d in self.data])
            alpha = 0.78 if self.spec.replay_mode == "aer" else 0.68
            prob = pri ** alpha
            prob /= prob.sum()
            idx = self.rng.choice(len(self.data), size=n, replace=False, p=prob)
        batch = [self.data[int(j)] for j in idx]
        self.sampled_count += len(batch)
        self.sampled_boundary += sum(int(x["is_boundary"]) for x in batch)
        self.sampled_uncertainty += sum(int(x["is_uncertainty"]) for x in batch)
        return batch

    def refresh_td_priorities(
        self, world: World, critic: LinearCritic, samples: Sequence[dict],
    ) -> None:
        """Refresh critic-dependent priority fields after a critic update."""
        if self.spec.replay_mode == "uniform":
            return
        for tr in samples:
            p = np.array([tr["x"], tr["y"]], dtype=float)
            p_next = np.array([tr["x_next"], tr["y_next"]], dtype=float)
            u_exec = np.array([tr["u_exec_x"], tr["u_exec_y"]], dtype=float)
            td = critic.td_error(
                world, p, u_exec, float(tr["instant_cost"]), p_next,
            )
            tr["td_error"] = td
            tr["td_score"] = min(1.0, abs(td) / 2.5)
            tr["priority"] = self._priority(tr)

    def audit(self, seed: int) -> dict:
        return {
            "method": self.spec.key,
            "short": self.spec.short,
            "seed": seed,
            "buffer_capacity": self.capacity,
            "final_buffer_size": len(self.data),
            "generated_boundary_fraction": self.generated_boundary / max(self.generated_count, 1),
            "stored_boundary_fraction": float(np.mean([x["is_boundary"] for x in self.data])) if self.data else 0.0,
            "sampled_boundary_fraction": self.sampled_boundary / max(self.sampled_count, 1),
            "generated_uncertainty_fraction": self.generated_uncertainty / max(self.generated_count, 1),
            "stored_uncertainty_fraction": float(np.mean([x["is_uncertainty"] for x in self.data])) if self.data else 0.0,
            "sampled_uncertainty_fraction": self.sampled_uncertainty / max(self.sampled_count, 1),
            "mean_priority": float(np.mean([x["priority"] for x in self.data])) if self.data else 0.0,
            "total_generated": self.generated_count,
            "total_sampled": self.sampled_count,
        }


@dataclass
class MethodState:
    spec: MethodSpec
    world: World
    seed: int
    actor: WaypointActor = field(init=False)
    critic: LinearCritic = field(default_factory=LinearCritic)
    estimator: OnlineCenterEstimator = field(init=False)
    replay: ReplayBuffer = field(init=False)
    visit_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actor = WaypointActor(self.world, self.seed)
        self.estimator = OnlineCenterEstimator(self.world, self.spec)
        self.replay = ReplayBuffer(360, self.spec, self.seed)


def clone_for_eval(state: MethodState) -> MethodState:
    clone = MethodState(state.spec, state.world, state.seed)
    clone.actor = copy.deepcopy(state.actor)
    clone.critic = copy.deepcopy(state.critic)
    clone.estimator = copy.deepcopy(state.estimator)
    clone.replay = ReplayBuffer(1, state.spec, state.seed)
    clone.visit_counts = copy.deepcopy(state.visit_counts)
    return clone


def estimated_barriers(
    world: World,
    p: np.ndarray,
    belief_centers: Sequence[np.ndarray],
    belief_sigmas: Sequence[float],
) -> List[Tuple[float, np.ndarray, str, float]]:
    barriers: List[Tuple[float, np.ndarray, str, float]] = []
    for r in world.rectangles:
        h, g = r.signed_distance_and_grad(p, inflation=world.robot_radius + 0.018)
        if h < 0.30:
            barriers.append((h, g, r.name, 0.0))
    for obs, center, sigma in zip(world.uncertain_obstacles, belief_centers, belief_sigmas):
        diff = p - center
        robust = 0.009 + 1.20 * sigma
        h = norm(diff) - (obs.radius + world.robot_radius + robust)
        if h < 0.40:
            barriers.append((h, unit(diff), obs.name + "_belief", robust))
    m = world.robot_radius + 0.018
    walls = [
        (p[0] - m, np.array([1.0, 0.0]), "world_left", 0.0),
        (world.width - m - p[0], np.array([-1.0, 0.0]), "world_right", 0.0),
        (p[1] - m, np.array([0.0, 1.0]), "world_bottom", 0.0),
        (world.height - m - p[1], np.array([0.0, -1.0]), "world_top", 0.0),
    ]
    barriers.extend([b for b in walls if b[0] < 0.26])
    return barriers


def minimum_belief_clearance(
    world: World,
    p: np.ndarray,
    centers: Sequence[np.ndarray],
    sigmas: Sequence[float],
) -> Tuple[float, np.ndarray, str]:
    barriers = estimated_barriers(world, p, centers, sigmas)
    if not barriers:
        return 1.0, np.zeros(2), "none"
    h, g, name, _ = min(barriers, key=lambda z: z[0])
    return float(h), g.copy(), name


def cbf_filter(
    world: World,
    p: np.ndarray,
    u_nom: np.ndarray,
    belief_centers: Sequence[np.ndarray],
    belief_sigmas: Sequence[float],
    use_cbf: bool,
    disturbance_bound: float,
) -> Tuple[np.ndarray, int, float, str, float, np.ndarray]:
    """Project the nominal velocity onto estimated CBF half-spaces.

    Uncertain-obstacle constraints use method-specific beliefs. Exact known-map
    constraints receive hard priority. The final discrete-time guard prevents a
    speed clipping or a competing uncertain-obstacle projection from creating a
    one-step penetration of a mapped rectangle/wall. No evaluator-only uncertain
    obstacle center is used.
    """
    min_h, min_grad, min_name = minimum_belief_clearance(
        world, p, belief_centers, belief_sigmas
    )
    if not use_cbf:
        return u_nom.copy(), 0, 0.0, min_name, min_h, min_grad

    barriers = estimated_barriers(world, p, belief_centers, belief_sigmas)
    known_names = {r.name for r in world.rectangles} | {
        "world_left", "world_right", "world_bottom", "world_top"
    }
    known_barriers = [b for b in barriers if b[2] in known_names]
    alpha = 2.9
    active = 0
    u = u_nom.astype(float).copy()

    def project_halfspaces(v: np.ndarray, selected, passes: int) -> np.ndarray:
        nonlocal active
        out = v.copy()
        for _ in range(passes):
            changed = False
            for h, g, _, robust in selected:
                if h >= 0.20:
                    continue
                rhs = -alpha * h + disturbance_bound + 0.18 * robust
                val = float(g @ out)
                if val < rhs:
                    out += (rhs - val) * g / max(float(g @ g), 1e-9)
                    active = 1
                    changed = True
            if not changed:
                break
        return out

    # Alternating projections reduce conflicts among nearby constraints. Known-map
    # projections are repeated last so a bad perception estimate cannot make the
    # robot cut through a mapped wall.
    for _ in range(4):
        u = project_halfspaces(u, barriers, 1)
        u = project_halfspaces(u, known_barriers, 2)
        speed = norm(u)
        if speed > world.max_speed:
            u *= world.max_speed / speed
        u = project_halfspaces(u, known_barriers, 2)

    speed = norm(u)
    if speed > world.max_speed:
        u *= world.max_speed / speed

    # Discrete-time known-map guard. It operates only on exact map geometry and
    # uses ``disturbance_bound`` as a design allowance.  The benchmark's
    # Gaussian AR(1) disturbance generator is not hard bounded, so this is not a
    # certified worst-case bound.
    guard = 0.0015 + world.dt * disturbance_bound
    for _ in range(10):
        q = p + world.dt * u
        h_next, _, g_next = world.known_static_clearance(q)
        if h_next >= guard:
            break
        needed = (guard - h_next) / max(world.dt, 1e-9) + disturbance_bound
        u += needed * g_next
        active = 1
        speed = norm(u)
        if speed > world.max_speed:
            u *= world.max_speed / speed

    # One final known-map half-space projection after the discrete guard.
    u = project_halfspaces(u, known_barriers, 3)
    speed = norm(u)
    if speed > world.max_speed:
        u *= world.max_speed / speed

    return u, active, norm(u - u_nom), min_name, min_h, min_grad

def replay_scores(
    estimated_h: float,
    correction: float,
    cbf_active: int,
    collision_observed: int,
    td_error: float,
    innovation_norms: Sequence[float],
    belief_sigmas: Sequence[float],
    observable_jump: Sequence[float],
    valid: Sequence[int],
    novelty_score: float,
) -> Tuple[float, float, float, float]:
    td_score = min(1.0, abs(td_error) / 2.5)
    safety_score = min(
        1.0,
        math.exp(-max(estimated_h, 0.0) / 0.075)
        + 0.34 * cbf_active + 0.30 * min(1.0, correction / 0.11)
        + 0.75 * collision_observed,
    )
    normalized_innovation = np.mean([
        inn / max(2.5 * sig, 0.030) for inn, sig in zip(innovation_norms, belief_sigmas)
    ])
    uncertainty_score = min(
        1.0,
        0.58 * normalized_innovation
        + 0.30 * float(np.mean(observable_jump))
        + 0.28 * (1.0 - float(np.mean(valid))),
    )
    return td_score, safety_score, uncertainty_score, min(1.0, max(0.0, novelty_score))


def replay_visible_transition(record: dict, n_obstacles: int) -> dict:
    """Controller-visible subset stored in replay; evaluator truth is excluded."""
    keys = {
        "method", "short", "label", "seed", "episode_id", "step", "time",
        "stress", "disturbance_stress", "train", "sensor_stream_hash",
        "x", "y", "x_next", "y_next", "u_nom_x", "u_nom_y", "u_exec_x", "u_exec_y",
        "exploration_x", "exploration_y", "actor_waypoint_index",
        "collision_observed", "cbf_active", "cbf_estimated_h", "estimated_active_constraint",
        "correction_norm", "instant_cost", "td_error", "td_score", "safety_score",
        "uncertainty_score", "novelty_score", "is_boundary", "is_uncertainty",
        "belief_grad_x", "belief_grad_y", "sensor_valid_fraction", "sensor_observable_jump_score",
    }
    for i in range(1, n_obstacles + 1):
        keys.update({
            f"obs{i}_raw_x", f"obs{i}_raw_y", f"obs{i}_belief_x", f"obs{i}_belief_y",
            f"obs{i}_sigma", f"obs{i}_reported_sigma", f"obs{i}_innovation",
            f"obs{i}_outlier_flag", f"obs{i}_valid", f"obs{i}_jump_score", f"obs{i}_radius",
        })
    return {key: record[key] for key in keys if key in record}


def transition_cost(
    world: World,
    p: np.ndarray,
    u_nom: np.ndarray,
    u_exec: np.ndarray,
    correction: float,
    known_h: float,
    collision_observed: int,
) -> float:
    goal_error = norm(p - world.goal)
    return float(
        0.036 * goal_error ** 2
        + 0.020 * norm(u_exec) ** 2
        + 0.012 * norm(u_exec - u_nom) ** 2
        + 0.036 * correction
        + 5.2 * collision_observed
        + 1.0 * max(0.0, 0.010 - known_h) ** 2
    )


def simulate_episode(
    state: MethodState,
    episode_id: int,
    stress: float,
    disturbance_stress: float,
    train: bool,
    record: bool,
) -> Tuple[dict, Optional[pd.DataFrame]]:
    world, spec, seed = state.world, state.spec, state.seed
    n_steps = world.train_horizon if train else world.eval_horizon
    sensor = generate_shared_sensor_stream(world, seed, episode_id, stress, n_steps)
    disturbance = generate_shared_disturbance_stream(seed, episode_id, disturbance_stress, n_steps)
    exploration_rng = np.random.default_rng(81_021 + 1000 * seed + 53 * episode_id)
    exploration = exploration_rng.normal(scale=0.017, size=(n_steps, 2))

    p = world.start.copy()
    state.actor.reset_episode()
    state.estimator.reset_episode(sensor.raw[0])
    rows: List[dict] = []
    total_cost = 0.0
    hs: List[float] = []
    known_hs: List[float] = []
    uncertain_hs: List[float] = []
    corrections: List[float] = []
    cbf_flags: List[int] = []
    raw_errors: List[float] = []
    belief_errors: List[float] = []
    belief_errors_each: List[List[float]] = [[] for _ in world.uncertain_obstacles]
    violations = known_violations = uncertain_violations = 0
    path = [p.copy()]
    reached_first_step: Optional[int] = None
    start_clock = time.perf_counter()

    for k in range(n_steps):
        raw = sensor.raw[k]
        belief_centers, belief_sigmas, innovations, outlier_flags = state.estimator.observe(
            raw, sensor.valid[k], sensor.reported_sigma[k], sensor.observable_jump[k]
        )
        if norm(p - world.goal) < world.goal_tolerance:
            if reached_first_step is None:
                reached_first_step = k
            u_nom = np.zeros(2)
        else:
            u_nom = state.actor.act(world, p, belief_centers, belief_sigmas, exploration[k], train=train)
        u_exec, cbf_active, correction, estimated_name, estimated_h, belief_grad = cbf_filter(
            world, p, u_nom, belief_centers, belief_sigmas, spec.use_cbf,
            disturbance_bound=0.012 * disturbance_stress,
        )
        p_next = p + world.dt * (u_exec + disturbance[k])
        p_next = np.clip(
            p_next,
            [world.robot_radius, world.robot_radius],
            [world.width - world.robot_radius, world.height - world.robot_radius],
        )
        if reached_first_step is None and norm(p_next - world.goal) < world.goal_tolerance:
            reached_first_step = k + 1

        # Physical/evaluator diagnostics. They do not enter belief, CBF, or replay priority.
        h_true, true_name, true_grad = world.true_clearance(p)
        h_known, known_name, _ = world.known_static_clearance(p)
        h_uncertain, uncertain_name, _ = world.uncertain_true_clearance(p)
        collision_observed = int(h_true < 0.0)  # an observable contact/bumper event
        instant_cost = transition_cost(world, p, u_nom, u_exec, correction, h_known, collision_observed)
        td_error = state.critic.td_error(world, p, u_exec, instant_cost, p_next)
        cell = (int(p[0] / 0.18), int(p[1] / 0.18))
        visits = state.visit_counts.get(cell, 0)
        observable_novelty = 1.0 / math.sqrt(1.0 + visits)
        state.visit_counts[cell] = visits + 1
        td_score, safety_score, uncertainty_score, novelty_score = replay_scores(
            estimated_h, correction, cbf_active, collision_observed, td_error,
            innovations, belief_sigmas, sensor.observable_jump[k], sensor.valid[k], observable_novelty,
        )
        is_boundary = int(estimated_h < 0.115 or cbf_active or correction > 0.020 or collision_observed)
        is_uncertainty = int(uncertainty_score > 0.48 or max(outlier_flags) > 0)

        raw_e = [norm(raw[i] - obs.c) for i, obs in enumerate(world.uncertain_obstacles)]
        belief_e = [norm(belief_centers[i] - obs.c) for i, obs in enumerate(world.uncertain_obstacles)]
        raw_rmse_step = float(np.sqrt(np.mean(np.square(raw_e))))
        belief_rmse_step = float(np.sqrt(np.mean(np.square(belief_e))))

        tr: Dict[str, object] = {
            "method": spec.key, "short": spec.short, "label": spec.label,
            "seed": seed, "episode_id": episode_id, "step": k, "time": k * world.dt,
            "stress": stress, "disturbance_stress": disturbance_stress, "train": int(train),
            "sensor_stream_hash": sensor.stream_hash,
            "x": p[0], "y": p[1], "x_next": p_next[0], "y_next": p_next[1],
            "u_nom_x": u_nom[0], "u_nom_y": u_nom[1],
            "u_exec_x": u_exec[0], "u_exec_y": u_exec[1],
            "exploration_x": exploration[k, 0] if train else 0.0,
            "exploration_y": exploration[k, 1] if train else 0.0,
            "actor_waypoint_index": state.actor.wp_idx,
            "disturbance_x": disturbance[k, 0], "disturbance_y": disturbance[k, 1],
            "true_clearance": h_true, "h": h_true, "true_active_constraint": true_name,
            "known_static_clearance": h_known, "known_static_active_constraint": known_name,
            "uncertain_true_clearance": h_uncertain, "uncertain_true_active_constraint": uncertain_name,
            "known_static_violation": int(h_known < 0.0),
            "uncertain_true_violation": int(h_uncertain < 0.0),
            "violation": collision_observed, "collision_observed": collision_observed,
            "cbf_active": cbf_active, "cbf_estimated_h": estimated_h,
            "estimated_active_constraint": estimated_name, "correction_norm": correction,
            "instant_cost": instant_cost, "td_error": td_error, "td_score": td_score,
            "safety_score": safety_score, "uncertainty_score": uncertainty_score,
            "novelty_score": novelty_score, "is_boundary": is_boundary, "is_uncertainty": is_uncertainty,
            "belief_grad_x": belief_grad[0], "belief_grad_y": belief_grad[1],
            "true_grad_x": true_grad[0], "true_grad_y": true_grad[1],
            "raw_center_rmse": raw_rmse_step, "belief_center_rmse": belief_rmse_step,
            "sensor_valid_fraction": float(np.mean(sensor.valid[k])),
            "sensor_observable_jump_score": float(np.mean(sensor.observable_jump[k])),
            "sensor_hidden_event": int(np.any(sensor.hidden_event[k] > 0)),
        }
        for i, obs in enumerate(world.uncertain_obstacles, start=1):
            bi = belief_centers[i - 1]
            tr.update({
                f"obs{i}_true_x": obs.c[0], f"obs{i}_true_y": obs.c[1],
                f"obs{i}_raw_x": raw[i - 1, 0], f"obs{i}_raw_y": raw[i - 1, 1],
                f"obs{i}_belief_x": bi[0], f"obs{i}_belief_y": bi[1],
                f"obs{i}_sigma": belief_sigmas[i - 1],
                f"obs{i}_reported_sigma": sensor.reported_sigma[k, i - 1],
                f"obs{i}_innovation": innovations[i - 1],
                f"obs{i}_outlier_flag": outlier_flags[i - 1],
                f"obs{i}_valid": sensor.valid[k, i - 1],
                f"obs{i}_jump_score": sensor.observable_jump[k, i - 1],
                f"obs{i}_hidden_event": sensor.hidden_event[k, i - 1],
                f"obs{i}_radius": obs.radius,
                f"obs{i}_raw_error": raw_e[i - 1],
                f"obs{i}_belief_error": belief_e[i - 1],
            })

        if train:
            state.replay.add(replay_visible_transition(tr, len(world.uncertain_obstacles)))
        if record:
            rows.append(tr)

        total_cost += instant_cost
        violations += collision_observed
        known_violations += int(h_known < 0.0)
        uncertain_violations += int(h_uncertain < 0.0)
        hs.append(h_true)
        known_hs.append(h_known)
        uncertain_hs.append(h_uncertain)
        corrections.append(correction)
        cbf_flags.append(cbf_active)
        raw_errors.append(raw_rmse_step)
        belief_errors.append(belief_rmse_step)
        for i, e in enumerate(belief_e):
            belief_errors_each[i].append(e)
        p = p_next
        path.append(p.copy())

    if train:
        for _ in range(8):
            batch = state.replay.sample(44)
            state.critic.update(world, batch)
            state.replay.refresh_td_priorities(world, state.critic, batch)
            state.actor.update_from_samples(world, batch, state.critic, lr=0.0040)
            state.estimator.replay_calibration(batch)

    runtime_ms = 1000.0 * (time.perf_counter() - start_clock)
    pts = np.asarray(path)
    goal_error_cm = 100.0 * norm(p - world.goal)
    result: Dict[str, object] = {
        "method": spec.key, "short": spec.short, "label": spec.label,
        "seed": seed, "episode_id": episode_id, "stress": stress,
        "disturbance_stress": disturbance_stress, "cost": total_cost,
        "sensor_stream_hash": sensor.stream_hash,
        "sensor_hold_fraction": float(1.0 - np.mean(sensor.valid)),
        "sensor_observable_jump_mean": float(np.mean(sensor.observable_jump)),
        "sensor_hidden_event_count": int(np.sum(sensor.hidden_event)),
        "violation_rate": violations / n_steps, "violation_count": violations,
        "known_static_violation_rate": known_violations / n_steps,
        "known_static_violation_count": known_violations,
        "uncertain_true_violation_rate": uncertain_violations / n_steps,
        "uncertain_true_violation_count": uncertain_violations,
        "min_clearance_cm": 100.0 * float(np.min(hs)),
        "min_known_static_clearance_cm": 100.0 * float(np.min(known_hs)),
        "min_uncertain_true_clearance_cm": 100.0 * float(np.min(uncertain_hs)),
        "mean_clearance_cm": 100.0 * float(np.mean(hs)),
        "goal_error_cm": goal_error_cm,
        "reached": int(reached_first_step is not None),
        "first_reach_step": -1 if reached_first_step is None else reached_first_step,
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))),
        "mean_cbf_correction": float(np.mean(corrections)),
        "cbf_intervention_rate": float(np.mean(cbf_flags)),
        "raw_sensor_rmse_cm": 100.0 * float(np.sqrt(np.mean(np.square(raw_errors)))),
        "belief_center_rmse_cm": 100.0 * float(np.sqrt(np.mean(np.square(belief_errors)))),
        "runtime_ms_per_episode": runtime_ms,
    }
    for i, vals in enumerate(belief_errors_each, start=1):
        result[f"obs{i}_belief_rmse_cm"] = 100.0 * float(np.sqrt(np.mean(np.square(vals))))
    return result, pd.DataFrame(rows) if record else None


# Reporting and visualization functions are appended below.

# -----------------------------------------------------------------------------
# Audits, tables, plots, animation, and packaging
# -----------------------------------------------------------------------------


def cfg_lookup() -> Dict[str, MethodSpec]:
    return {s.key: s for s in method_specs()}


def _order(df: pd.DataFrame) -> pd.DataFrame:
    if "method" not in df.columns:
        return df
    out = df.copy()
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    out["_method_order"] = out["method"].map(order)
    sort_cols = [c for c in ["_method_order", "seed", "epoch", "step", "stress_multiplier", "waypoint"] if c in out.columns]
    out = out.sort_values(sort_cols).drop(columns="_method_order")
    return out


def write_sensor_fairness_audit(out_dir: Path) -> None:
    rows: List[dict] = []
    for eval_type, filename in [("nominal", "nominal_eval_metrics.csv"), ("moderate", "moderate_eval_metrics.csv"), ("stress", "stress_eval_metrics.csv")]:
        df = pd.read_csv(out_dir / filename)
        for seed, g in df.groupby("seed"):
            rows.append({
                "eval_type": eval_type,
                "seed": int(seed),
                "methods_present": int(g.method.nunique()),
                "unique_sensor_stream_hashes": int(g.sensor_stream_hash.nunique()),
                "raw_sensor_rmse_range_cm": float(g.raw_sensor_rmse_cm.max() - g.raw_sensor_rmse_cm.min()),
                "sensor_hold_fraction_range": float(g.sensor_hold_fraction.max() - g.sensor_hold_fraction.min()),
                "sensor_jump_mean_range": float(g.sensor_observable_jump_mean.max() - g.sensor_observable_jump_mean.min()),
                "pass_identical_common_sensor": int(
                    g.method.nunique() == len(METHOD_ORDER)
                    and g.sensor_stream_hash.nunique() == 1
                    and abs(g.raw_sensor_rmse_cm.max() - g.raw_sensor_rmse_cm.min()) < 1e-12
                    and abs(g.sensor_hold_fraction.max() - g.sensor_hold_fraction.min()) < 1e-12
                    and abs(g.sensor_observable_jump_mean.max() - g.sensor_observable_jump_mean.min()) < 1e-12
                ),
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "sensor_fairness_audit.csv", index=False, encoding="utf-8")

    traj = pd.read_csv(out_dir / "evaluation_trajectories_seed0.csv")
    ts_rows: List[dict] = []
    for eval_type, d_eval in traj.groupby("eval_type"):
        for obs_i in (1, 2):
            for step, g in d_eval.groupby("step"):
                xs = g[f"obs{obs_i}_raw_x"].to_numpy(dtype=float)
                ys = g[f"obs{obs_i}_raw_y"].to_numpy(dtype=float)
                valid = g[f"obs{obs_i}_valid"].to_numpy(dtype=int)
                ts_rows.append({
                    "eval_type": eval_type,
                    "obstacle": obs_i,
                    "step": int(step),
                    "method_count": int(g.method.nunique()),
                    "max_raw_x_difference_across_methods": float(xs.max() - xs.min()),
                    "max_raw_y_difference_across_methods": float(ys.max() - ys.min()),
                    "valid_identical_across_methods": int(np.all(valid == valid[0])),
                })
    ts = pd.DataFrame(ts_rows)
    ts.to_csv(out_dir / "sensor_fairness_seed0_timeseries.csv", index=False, encoding="utf-8")

    passed = bool(audit.pass_identical_common_sensor.eq(1).all())
    pointwise = bool(
        ts.method_count.eq(len(METHOD_ORDER)).all()
        and ts.max_raw_x_difference_across_methods.abs().max() < 1e-12
        and ts.max_raw_y_difference_across_methods.abs().max() < 1e-12
        and ts.valid_identical_across_methods.eq(1).all()
    )
    lines = [
        "# Shared-sensor fairness audit",
        "",
        f"**Summary status: {'PASS' if passed and pointwise else 'FAIL'}**",
        "",
        "For each seed/evaluation, all six methods must have one common SHA-256 sensor-stream hash, identical raw-sensor RMSE, identical hold fraction, and identical observable jump statistics.",
        "The seed-0 pointwise audit additionally requires exactly zero x/y measurement spread across methods at every step.",
        "",
        audit.to_markdown(index=False),
        "",
        f"Pointwise seed-0 equality: {'PASS' if pointwise else 'FAIL'}.",
    ]
    (out_dir / "sensor_fairness_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_oracle_invariance_audit(out_dir: Path) -> None:
    """Black-box check that hidden true-center translation cannot change control."""
    rows: List[dict] = []
    p = np.array([0.96, 1.68], dtype=float)
    beliefs = [np.array([0.76, 1.52]), np.array([1.89, 1.02])]
    sigmas = [0.024, 0.048]
    exploration = np.zeros(2)
    for spec in method_specs():
        wa, wb = World(), World()
        wb.uncertain_obstacles = [
            CircleObstacle(o.name, (o.center[0] + 0.43, o.center[1] - 0.31), o.radius)
            for o in wb.uncertain_obstacles
        ]
        aa, ab = WaypointActor(wa, seed=17), WaypointActor(wb, seed=17)
        ua = aa.act(wa, p, beliefs, sigmas, exploration, train=False)
        ub = ab.act(wb, p, beliefs, sigmas, exploration, train=False)
        ea, *_ = cbf_filter(wa, p, ua, beliefs, sigmas, spec.use_cbf, disturbance_bound=0.012)
        eb, *_ = cbf_filter(wb, p, ub, beliefs, sigmas, spec.use_cbf, disturbance_bound=0.012)
        nom_diff = float(np.max(np.abs(ua - ub)))
        exec_diff = float(np.max(np.abs(ea - eb)))
        rows.append({
            "method": spec.key,
            "hidden_truth_translation_x_m": 0.43,
            "hidden_truth_translation_y_m": -0.31,
            "max_nominal_action_difference": nom_diff,
            "max_executed_action_difference": exec_diff,
            "pass_no_oracle_dependence": int(nom_diff < 1e-12 and exec_diff < 1e-12),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "oracle_invariance_audit.csv", index=False, encoding="utf-8")
    lines = [
        "# Hidden-truth oracle invariance audit",
        "",
        "The evaluator-only true centers are translated while all controller-visible beliefs, uncertainty radii, state, and known-map geometry are held fixed.",
        "Nominal and executed actions must remain unchanged.",
        "",
        audit.to_markdown(index=False),
        "",
        f"**Status: {'PASS' if audit.pass_no_oracle_dependence.eq(1).all() else 'FAIL'}**",
    ]
    (out_dir / "oracle_invariance_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(vals: pd.Series, decimals: int = 2, percent: bool = False) -> str:
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    if percent:
        return f"{100.0 * mean:.{decimals}f} ± {100.0 * std:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def make_tables(out_dir: Path) -> None:
    stress = pd.read_csv(out_dir / "stress_eval_metrics.csv")
    moderate_path = out_dir / "moderate_eval_metrics.csv"
    moderate = pd.read_csv(moderate_path) if moderate_path.exists() else None
    replay = pd.read_csv(out_dir / "replay_finite_buffer_audit.csv")
    specs = method_specs()
    metrics = [
        ("cost", "Cost ↓", False, "down", 2),
        ("violation_rate", "Violation rate (%) ↓", True, "down", 2),
        ("min_clearance_cm", "Minimum true clearance (cm) ↑", False, "up", 2),
        ("min_uncertain_true_clearance_cm", "Minimum true uncertain-obstacle clearance (cm) ↑", False, "up", 2),
        ("belief_center_rmse_cm", "Obstacle-belief RMSE (cm) ↓", False, "down", 2),
        ("goal_error_cm", "Goal error (cm) ↓", False, "down", 2),
        ("reached", "Goal success (%) ↑", True, "up", 1),
        ("cbf_intervention_rate", "CBF intervention (%) ↓", True, "down", 2),
        ("runtime_ms_per_episode", "Runtime/episode (ms) ↓", False, "down", 2),
    ]

    def build_eval_table(df: pd.DataFrame, base: str, heading: str, caption: str, label: str) -> None:
        rows: List[dict] = []
        for spec in specs:
            d = df[df.method == spec.key]
            row: Dict[str, str] = {"Method": spec.short}
            for col, display, pct, _, dec in metrics:
                if col == "cbf_intervention_rate" and not spec.use_cbf:
                    row[display] = "N/A"
                else:
                    row[display] = _fmt(d[col], dec, pct)
            rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(out_dir / f"{base}.csv", index=False, encoding="utf-8")

        best: Dict[str, set] = {}
        for col, display, _, direction, _ in metrics:
            means: Dict[str, float] = {}
            for spec in specs:
                if col == "cbf_intervention_rate" and not spec.use_cbf:
                    continue
                means[spec.short] = float(df[df.method == spec.key][col].mean())
            optimum = min(means.values()) if direction == "down" else max(means.values())
            best[display] = {m for m, v in means.items() if abs(v - optimum) < 1e-12}

        md = [
            f"# {heading} (mean ± standard deviation across seeds)",
            "",
            "| " + " | ".join(table.columns) + " |",
            "| " + " | ".join(["---"] * len(table.columns)) + " |",
        ]
        for _, row in table.iterrows():
            vals = [str(row["Method"])]
            for display in table.columns[1:]:
                val = str(row[display])
                if row["Method"] in best[display]:
                    val = f"**{val}**"
                vals.append(val)
            md.append("| " + " | ".join(vals) + " |")
        md += [
            "",
            "Raw sensor-center RMSE is omitted because the raw stream is exactly identical across methods; see `sensor_fairness_audit.csv`.",
            "`AC` has no CBF, so its intervention rate is not applicable. All tied mean optima are bolded.",
            "Runtime is platform-specific and is not an algorithm-independent complexity bound.",
        ]
        (out_dir / f"{base}.md").write_text("\n".join(md), encoding="utf-8")

        headers = [
            c.replace("%", "\\%").replace("↓", "$\\downarrow$").replace("↑", "$\\uparrow$")
            for c in table.columns
        ]
        tex = [
            "\\begin{table*}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{" + "l" + "c" * (len(table.columns) - 1) + "}",
            "\\toprule",
            " & ".join(headers) + " \\\\",
            "\\midrule",
        ]
        for _, row in table.iterrows():
            vals = [str(row["Method"])]
            for display in table.columns[1:]:
                val = str(row[display]).replace("±", "$\\pm$")
                if row["Method"] in best[display]:
                    val = "\\textbf{" + val + "}"
                vals.append(val)
            tex.append(" & ".join(vals) + " \\\\")
        tex += [
            "\\bottomrule", "\\end{tabular}",
            "\\begin{minipage}{0.99\\textwidth}\\footnotesize\\vspace{2pt}",
            "AC has no CBF, so its intervention rate is N/A. Raw sensor-center RMSE is common and appears in the fairness audit.",
            "\\end{minipage}", "\\end{table*}",
        ]
        (out_dir / f"{base}.tex").write_text("\n".join(tex), encoding="utf-8")

    build_eval_table(
        stress,
        "paper_ready_evaluation_table",
        "Severe perception-stress results",
        "Severe perception-stress results (mean $\\pm$ standard deviation over five seeds). The raw sensor stream is identical across methods.",
        "tab:v17_1_severe_results",
    )
    if moderate is not None:
        build_eval_table(
            moderate,
            "paper_ready_moderate_evaluation_table",
            "Moderate perception-stress results",
            "Moderate perception-stress results (mean $\\pm$ standard deviation over five seeds). The raw sensor stream is identical across methods.",
            "tab:v17_1_moderate_results",
        )

    replay_cols = [
        "generated_boundary_fraction", "stored_boundary_fraction", "sampled_boundary_fraction",
        "generated_uncertainty_fraction", "stored_uncertainty_fraction", "sampled_uncertainty_fraction",
    ]
    rr: List[dict] = []
    for spec in specs:
        g = replay[replay.method == spec.key]
        row: Dict[str, str] = {"Method": spec.short}
        for col in replay_cols:
            row[col] = _fmt(g[col], 3, False)
        rr.append(row)
    rtable = pd.DataFrame(rr)
    rtable.to_csv(out_dir / "paper_ready_replay_audit_table.csv", index=False, encoding="utf-8")
    (out_dir / "paper_ready_replay_audit_table.md").write_text(
        rtable.to_markdown(index=False), encoding="utf-8"
    )
    (out_dir / "paper_ready_replay_audit_table.tex").write_text(
        rtable.to_latex(index=False, escape=True), encoding="utf-8"
    )

    full = stress[stress.method == "Full"].sort_values("seed")
    stat_rows: List[dict] = []
    for spec in specs:
        if spec.key == "Full":
            continue
        d = stress[stress.method == spec.key].sort_values("seed")
        for metric in ["cost", "violation_rate", "min_clearance_cm", "belief_center_rmse_cm"]:
            x, y = d[metric].to_numpy(), full[metric].to_numpy()
            if SCIPY_AVAILABLE and len(x) == len(y):
                try:
                    if np.allclose(x, y, rtol=0.0, atol=1e-15):
                        statistic, pvalue = 0.0, 1.0
                    else:
                        res = wilcoxon(
                            x, y, zero_method="wilcox", alternative="two-sided",
                            method="auto",
                        )
                        statistic, pvalue = float(res.statistic), float(res.pvalue)
                except Exception:
                    statistic, pvalue = float("nan"), float("nan")
            else:
                statistic, pvalue = float("nan"), float("nan")
            stat_rows.append({
                "comparison": f"{spec.short} vs Full",
                "metric": metric,
                "statistic": statistic,
                "p_value": pvalue,
            })
    pd.DataFrame(stat_rows).to_csv(
        out_dir / "stress_wilcoxon_vs_full.csv", index=False, encoding="utf-8"
    )

    friedman_rows: List[dict] = []
    friedman_metrics = [
        "cost", "violation_rate", "min_clearance_cm",
        "belief_center_rmse_cm", "goal_error_cm",
    ]
    if SCIPY_AVAILABLE:
        for metric in friedman_metrics:
            arrays = [
                stress[stress.method == spec.key].sort_values("seed")[metric].to_numpy()
                for spec in specs
            ]
            result = friedmanchisquare(*arrays)
            friedman_rows.append({
                "scope": "all_six_methods",
                "metric": metric,
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "method_count": len(arrays),
                "seed_count": len(arrays[0]),
            })
        cbf_specs = [spec for spec in specs if spec.use_cbf]
        arrays = [
            stress[stress.method == spec.key].sort_values("seed")["cbf_intervention_rate"].to_numpy()
            for spec in cbf_specs
        ]
        result = friedmanchisquare(*arrays)
        friedman_rows.append({
            "scope": "five_cbf_methods",
            "metric": "cbf_intervention_rate",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method_count": len(arrays),
            "seed_count": len(arrays[0]),
        })
    pd.DataFrame(friedman_rows).to_csv(
        out_dir / "stress_friedman_tests.csv", index=False, encoding="utf-8"
    )

def _add_world(ax: plt.Axes, world: World, labels: bool = True) -> None:
    for r in world.rectangles:
        ax.add_patch(Rectangle((100 * r.xmin, 100 * r.ymin), 100 * (r.xmax - r.xmin), 100 * (r.ymax - r.ymin), facecolor="#aaaaaa", edgecolor="#666666", linewidth=1.2, zorder=1))
    for obs in world.uncertain_obstacles:
        ax.add_patch(Circle((100 * obs.c[0], 100 * obs.c[1]), 100 * (obs.radius + world.robot_radius), fill=False, edgecolor="#b31b1b", linewidth=2.0, zorder=3))
        ax.add_patch(Circle((100 * obs.c[0], 100 * obs.c[1]), 100 * obs.radius, facecolor="#ffdede", edgecolor="#b31b1b", linewidth=1.5, zorder=2))
    ax.scatter([100 * world.start[0]], [100 * world.start[1]], marker="s", s=190, facecolor="#00ee22", edgecolor="black", linewidth=1.4, zorder=20)
    ax.scatter([100 * world.goal[0]], [100 * world.goal[1]], marker="*", s=420, facecolor="#ff2828", edgecolor="black", linewidth=1.4, zorder=20)
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 320)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (cm)", fontweight="bold")
    ax.set_ylabel("y (cm)", fontweight="bold")
    ax.grid(True, alpha=0.24)
    if labels:
        ax.text(100 * world.start[0] + 4, 100 * world.start[1] + 4, "start", fontsize=9)
        ax.text(100 * world.goal[0] + 4, 100 * world.goal[1] + 4, "goal", fontsize=9)


def _savefig(fig: plt.Figure, path: Path, bottom: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bottom > 0:
        fig.subplots_adjust(bottom=bottom)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bar_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path, direction: str) -> None:
    specs = method_specs()
    means = np.array([df[df.method == s.key][metric].mean() for s in specs])
    stds = np.array([df[df.method == s.key][metric].std(ddof=1) for s in specs])
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(specs))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=[s.color for s in specs], edgecolor="black", linewidth=0.7)
    best = np.nanmin(means) if direction == "down" else np.nanmax(means)
    for b, value in zip(bars, means):
        weight = "bold" if abs(value - best) < 1e-12 else "normal"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9, fontweight=weight)
    ax.set_xticks(x, [s.short for s in specs], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, path)


def make_plots(out_dir: Path, display_time_s: float = 10.0) -> None:
    world = World()
    specs = method_specs()
    cfg = cfg_lookup()
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    display_files = sorted(out_dir.glob("evaluation_trajectories_display_seed*.csv"))
    trajectory_file = display_files[0] if display_files else out_dir / "evaluation_trajectories_seed0.csv"
    traj = pd.read_csv(trajectory_file)
    stress_t = traj[traj.eval_type == "stress"]
    display_seed = int(stress_t.seed.iloc[0])
    stress = pd.read_csv(out_dir / "stress_eval_metrics.csv")
    train = pd.read_csv(out_dir / "training_episode_metrics.csv")
    replay = pd.read_csv(out_dir / "replay_finite_buffer_audit.csv")
    sweep = pd.read_csv(out_dir / "robustness_sweep.csv")
    wp = pd.read_csv(out_dir / "learned_waypoints.csv")

    # Fig. 1: all stress trajectories plus shared raw sensor trace.
    fig, ax = plt.subplots(figsize=(10.0, 10.0))
    _add_world(ax, world, labels=False)
    raw_ref = stress_t[stress_t.method == "AC"]
    for i in (1, 2):
        ax.plot(100 * raw_ref[f"obs{i}_raw_x"], 100 * raw_ref[f"obs{i}_raw_y"], linestyle=":", linewidth=1.2, color="#d8a600", alpha=0.85, zorder=4)
        idx = np.arange(0, len(raw_ref), max(1, len(raw_ref) // 45))
        ax.scatter(100 * raw_ref.iloc[idx][f"obs{i}_raw_x"], 100 * raw_ref.iloc[idx][f"obs{i}_raw_y"], marker="x", s=25, color="#d8a600", zorder=5)
    for spec in specs:
        d = stress_t[stress_t.method == spec.key]
        ax.plot(100 * d.x, 100 * d.y, color=spec.color, linewidth=spec.line_width, label=spec.short, zorder=7)
        unsafe = d[d.violation > 0]
        if not unsafe.empty:
            ax.scatter(100 * unsafe.x, 100 * unsafe.y, marker="x", s=80, linewidth=2.2, color=spec.color, zorder=12)
    ax.set_title("Extreme perception-stress trajectories", fontsize=13, fontweight="bold")
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#aaaaaa", edgecolor="#666666", label="known static obstacle"),
        Line2D([0], [0], marker="s", markersize=11, markerfacecolor="#00ee22", markeredgecolor="black", linestyle="None", label="Start"),
        Line2D([0], [0], marker="*", markersize=16, markerfacecolor="#ff2828", markeredgecolor="black", linestyle="None", label="Goal"),
        Circle((0, 0), 1, fill=False, edgecolor="#b31b1b", linewidth=2, label="true robot-collision boundary"),
        Circle((0, 0), 1, facecolor="#ffdede", edgecolor="#b31b1b", label="true fixed uncertain obstacle"),
        Line2D([0], [0], color="#d8a600", linestyle=":", label="shared raw sensor-center trace"),
        Line2D([0], [0], marker="x", color="#d8a600", linestyle="None", label="shared raw sensor samples"),
        Line2D([0], [0], marker="x", color="#7a3a13", markeredgewidth=2, linestyle="None", label="unsafe robot sample (h<0)"),
    ] + [Line2D([0], [0], color=s.color, linewidth=s.line_width, label=s.short) for s in specs]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=True, fontsize=9)
    _savefig(fig, figs / "fig01_full_trajectory_comparison.png", bottom=0.18)

    # Fig. 2: Full trajectory, raw trace, UE-only belief, Full belief, waypoints.
    fig, ax = plt.subplots(figsize=(10.0, 9.5))
    _add_world(ax, world, labels=False)
    full = stress_t[stress_t.method == "Full"]
    ue = stress_t[stress_t.method == "AC+CBF+UE"]
    ax.plot(100 * full.x, 100 * full.y, color=cfg["Full"].color, linewidth=4.0, label="Full robot trajectory", zorder=8)
    w0 = wp[(wp.method == "Full") & (wp.seed == display_seed)].sort_values("waypoint")
    ax.plot(100 * w0.x, 100 * w0.y, color=cfg["Full"].color, linestyle=":", linewidth=1.7, label="learned actor waypoints")
    for i in (1, 2):
        ax.plot(100 * raw_ref[f"obs{i}_raw_x"], 100 * raw_ref[f"obs{i}_raw_y"], color="#d8a600", linestyle=":", linewidth=1.4, label="shared raw/no-UE belief" if i == 1 else None)
        ax.plot(100 * ue[f"obs{i}_belief_x"], 100 * ue[f"obs{i}_belief_y"], color=cfg["AC+CBF+UE"].color, linestyle="--", linewidth=1.5, label="UE-only filtered belief" if i == 1 else None)
        ax.plot(100 * full[f"obs{i}_belief_x"], 100 * full[f"obs{i}_belief_y"], color=cfg["Full"].color, linestyle="-.", linewidth=1.8, label="Full replay-calibrated belief" if i == 1 else None)
        for j in np.linspace(0, len(full) - 1, 8).astype(int):
            cx, cy = 100 * full.iloc[j][f"obs{i}_belief_x"], 100 * full.iloc[j][f"obs{i}_belief_y"]
            rad = 100 * (world.uncertain_obstacles[i-1].radius + world.robot_radius + 0.009 + 1.20 * full.iloc[j][f"obs{i}_sigma"])
            ax.add_patch(Circle((cx, cy), rad, fill=False, edgecolor=cfg["Full"].color, alpha=0.20, linewidth=1.0, linestyle="--"))
    ax.set_title("Full method: true fixed obstacles, shared raw observations, and learned obstacle beliefs", fontsize=15, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9, frameon=True)
    _savefig(fig, figs / "fig02_full_method_trajectory.png", bottom=0.13)

    # Fig. 3: true safety margins during the informative active-control interval.
    # The registered evaluation still runs for 280 transitions (15.4 s); only
    # the post-goal dwell tail is omitted from this visualization.
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for spec in specs:
        d = stress_t[(stress_t.method == spec.key) & (stress_t.time <= display_time_s)]
        ax.plot(d.time, 100 * d.true_clearance, color=spec.color, linewidth=spec.line_width, label=spec.short)
    ax.axhline(0.0, color="black", linewidth=1.6)
    ax.fill_between([0, display_time_s], -5, 0, color="#eeeeee", zorder=-2)
    ax.set_xlim(0, display_time_s)
    ax.set_ylim(-5, 30)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("minimum true signed clearance h(t) (cm)")
    ax.set_title(f"True safety margin under the common stress sensor stream (display seed {display_seed})", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    _savefig(fig, figs / "fig03_safety_margin_over_time.png")

    _bar_metric(stress, "violation_rate", "Violation rate ↓", "Stress-test violation-rate ablation", figs / "fig04_stress_violation_ablation.png", "down")
    _bar_metric(stress, "min_uncertain_true_clearance_cm", "Minimum true uncertain-obstacle clearance (cm) ↑", "Stress-test uncertain-obstacle clearance ablation", figs / "fig05_stress_min_clearance_ablation.png", "up")

    # Fig. 5b: per-seed global minimum.
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for i, spec in enumerate(specs):
        vals = stress[stress.method == spec.key].sort_values("seed").min_clearance_cm.to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, s=55, color=spec.color, edgecolor="black", linewidth=0.4, zorder=3)
        ax.hlines(vals.min(), i - 0.28, i + 0.28, color=spec.color, linewidth=3)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xticks(range(len(specs)), [s.short for s in specs], rotation=18, ha="right")
    ax.set_ylabel("per-seed minimum true clearance (cm)")
    ax.set_title("Global stress-test minimum-clearance audit across seeds", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, figs / "fig05b_global_min_clearance_audit.png")

    _bar_metric(stress, "cost", "Episode cost ↓", "Stress-test cost ablation", figs / "fig06_stress_cost_ablation.png", "down")

    # Fig. 7 learning curves.
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for spec in specs:
        g = train[train.method == spec.key].groupby("epoch").cost.agg(["mean", "std"])
        x = g.index.to_numpy()
        ax.plot(x, g["mean"], color=spec.color, linewidth=spec.line_width, label=spec.short)
        ax.fill_between(x, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=spec.color, alpha=0.10)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("episode cost")
    ax.set_title("Training cost under the increasing-severity curriculum", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    _savefig(fig, figs / "fig07_learning_curve_cost.png")

    # Fig. 8 finite-buffer audit.
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    agg = replay.groupby("method").mean(numeric_only=True)
    x = np.arange(len(specs)); width = 0.12
    cols = [
        ("generated_boundary_fraction", "boundary generated"),
        ("stored_boundary_fraction", "boundary stored"),
        ("sampled_boundary_fraction", "boundary sampled"),
        ("generated_uncertainty_fraction", "uncertainty generated"),
        ("stored_uncertainty_fraction", "uncertainty stored"),
        ("sampled_uncertainty_fraction", "uncertainty sampled"),
    ]
    for j, (col, label) in enumerate(cols):
        vals = [agg.loc[s.key, col] for s in specs]
        ax.bar(x + (j - 2.5) * width, vals, width=width, label=label)
    ax.set_xticks(x, [s.short for s in specs], rotation=18, ha="right")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Finite-buffer replay composition and sampling audit", fontweight="bold")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, figs / "fig08_replay_finite_buffer_audit.png")

    # Figs. 9-10 robustness sweeps.
    for metric, ylabel, fname, title in [
        ("uncertain_true_violation_rate_mean", "True uncertain-obstacle violation rate ↓", "fig09_robustness_sweep_violation.png", "Hidden-obstacle safety under increasing sensor-position uncertainty"),
        ("min_uncertain_true_clearance_cm_mean", "Minimum true uncertain-obstacle clearance (cm) ↑", "fig10_robustness_sweep_clearance.png", "Hidden-obstacle clearance under increasing sensor-position uncertainty"),
    ]:
        fig, ax = plt.subplots(figsize=(10.5, 5.7))
        for spec in specs:
            d = sweep[sweep.method == spec.key].sort_values("stress_multiplier")
            y = d[metric].to_numpy()
            std_col = metric.replace("_mean", "_std")
            e = d[std_col].fillna(0).to_numpy()
            ax.plot(d.stress_multiplier, y, marker="o", color=spec.color, linewidth=spec.line_width, label=spec.short)
            lower = np.maximum(0.0, y - e) if "violation" in metric else y - e
            ax.fill_between(d.stress_multiplier, lower, y + e, color=spec.color, alpha=0.08)
        if "violation" in metric:
            ax.set_ylim(bottom=-0.001)
        ax.set_xlabel("sensor-position uncertainty multiplier")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=9)
        _savefig(fig, figs / fname)

    # Fig. 11 belief error over time.
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    raw = stress_t[stress_t.method == "AC"]
    ue = stress_t[stress_t.method == "AC+CBF+UE"]
    full = stress_t[stress_t.method == "Full"]
    ax.plot(raw.time, 100 * raw.raw_center_rmse, color="#d8a600", linewidth=2.0, label="shared raw/no-UE belief")
    ax.plot(ue.time, 100 * ue.belief_center_rmse, color=cfg["AC+CBF+UE"].color, linewidth=2.3, label="UE-only belief")
    ax.plot(full.time, 100 * full.belief_center_rmse, color=cfg["Full"].color, linewidth=3.2, label="Full replay-calibrated belief")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("obstacle-center belief RMSE (cm)")
    ax.set_title(f"Estimator accuracy under the same jumping sensor measurements (display seed {display_seed})", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    _savefig(fig, figs / "fig11_obstacle_belief_rmse_over_time.png")

    # Fig. 12 time-sequence panels.
    snapshot_steps = [50, 75, 145, 205]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.4), sharex=True, sharey=True)
    for ax, step in zip(axes, snapshot_steps):
        _add_world(ax, world, labels=False)
        for spec in specs:
            d = stress_t[(stress_t.method == spec.key) & (stress_t.step <= step)]
            ax.plot(100 * d.x, 100 * d.y, color=spec.color, linewidth=1.5, alpha=0.85)
            if not d.empty:
                ax.scatter(100 * d.iloc[-1].x, 100 * d.iloc[-1].y, color=spec.color, s=24, zorder=15)
        ref = raw_ref[raw_ref.step == step]
        ue_row = ue[ue.step == step]
        full_row = full[full.step == step]
        if not ref.empty:
            for i, obs in enumerate(world.uncertain_obstacles, start=1):
                rr = ref.iloc[0]
                ax.scatter(100 * rr[f"obs{i}_raw_x"], 100 * rr[f"obs{i}_raw_y"], marker="x", color="#d8a600", s=65, linewidth=2, zorder=20)
                if not ue_row.empty:
                    ur = ue_row.iloc[0]
                    ax.scatter(100 * ur[f"obs{i}_belief_x"], 100 * ur[f"obs{i}_belief_y"], marker="o", facecolors="none", edgecolors=cfg["AC+CBF+UE"].color, s=70, linewidth=1.8, zorder=20)
                if not full_row.empty:
                    fr = full_row.iloc[0]
                    ax.scatter(100 * fr[f"obs{i}_belief_x"], 100 * fr[f"obs{i}_belief_y"], marker="s", color=cfg["Full"].color, s=42, zorder=20)
        ax.set_title(f"t = {step * world.dt:.2f} s", fontsize=11, fontweight="bold")
        ax.set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")
    fig.suptitle("Time sequence: fixed truth, shared raw observation, and method beliefs", fontsize=14, fontweight="bold")
    sequence_handles = [
        Circle((0, 0), 1, facecolor="#ffdede", edgecolor="#b31b1b", label="fixed true obstacle"),
        Line2D([0], [0], marker="x", color="#d8a600", markeredgewidth=2, linestyle="None", label="shared raw / no-UE belief"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=cfg["AC+CBF+UE"].color, markeredgewidth=2, linestyle="None", label="UE-only belief"),
        Line2D([0], [0], marker="s", color=cfg["Full"].color, linestyle="None", label="Full belief"),
    ]
    fig.legend(handles=sequence_handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=True, fontsize=9)
    _savefig(fig, figs / "fig12_time_sequence_panels.png", bottom=0.15)

    _bar_metric(stress, "belief_center_rmse_cm", "Obstacle-belief RMSE (cm) ↓", "Stress-test obstacle-center estimation ablation", figs / "fig13_belief_rmse_ablation.png", "down")
    make_plot_gallery(out_dir)


def make_plot_gallery(out_dir: Path) -> None:
    images = sorted((out_dir / "figures").glob("*.png"))
    cards = "\n".join(
        f'<figure><img src="figures/{p.name}" alt="{p.stem}"><figcaption>{p.stem}</figcaption></figure>'
        for p in images
    )
    html = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>V17.1 plot gallery</title>
<style>body{{font-family:Arial;margin:20px;background:#fafafa}}figure{{background:white;border:1px solid #ddd;padding:10px;margin:16px auto;max-width:1100px}}img{{width:100%;height:auto}}figcaption{{font-weight:bold;margin-top:6px}}</style></head><body><h1>V17.1 plot gallery</h1>{cards}</body></html>"""
    (out_dir / "plot_gallery.html").write_text(html, encoding="utf-8")


def make_animation(
    out_dir: Path,
    eval_type: str = "stress",
    frames: int = 90,
    fps: int = 9,
    display_time_s: float = 10.0,
) -> None:
    """Create an active-interval animation aligned with the trajectory CSV."""
    world = World()
    specs = method_specs()
    cfg = cfg_lookup()
    display_files = sorted(out_dir.glob("evaluation_trajectories_display_seed*.csv"))
    trajectory_file = display_files[0] if display_files else out_dir / "evaluation_trajectories_seed0.csv"
    traj = pd.read_csv(trajectory_file)
    d_all = traj[(traj.eval_type == eval_type) & (traj.time <= display_time_s)].copy()
    if d_all.empty:
        raise ValueError(f"no trajectory samples at or before {display_time_s:g} s")
    display_seed = int(d_all.seed.iloc[0])
    n_steps = int(d_all.step.max()) + 1
    display_end_s = float(d_all.time.max())
    frame_steps = np.linspace(0, n_steps - 1, frames).astype(int)
    methods = {s.key: d_all[d_all.method == s.key].sort_values("step").reset_index(drop=True) for s in specs}
    raw = methods["AC"]
    ue = methods["AC+CBF+UE"]
    full = methods["Full"]

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.18], height_ratios=[1, 1], left=0.04, right=0.97, top=0.86, bottom=0.17, wspace=0.13, hspace=0.29)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1])
    _add_world(ax_map, world, labels=False)
    ax_map.set_title("Map: fixed truth, shared raw measurement, and method beliefs", fontsize=12, fontweight="bold", pad=10)

    path_lines: Dict[str, Line2D] = {}
    robot_dots: Dict[str, Line2D] = {}
    unsafe_lines: Dict[str, Line2D] = {}
    h_lines: Dict[str, Line2D] = {}
    for spec in specs:
        path_lines[spec.key], = ax_map.plot([], [], color=spec.color, linewidth=spec.line_width, alpha=0.95)
        robot_dots[spec.key], = ax_map.plot([], [], marker="o", markersize=7 if spec.key != "Full" else 9, color=spec.color, markeredgecolor="black", linestyle="None", zorder=25)
        unsafe_lines[spec.key], = ax_map.plot([], [], marker="x", markersize=7, markeredgewidth=2, color=spec.color, linestyle="None", zorder=26)
        h_lines[spec.key], = ax_h.plot([], [], color=spec.color, linewidth=spec.line_width, label=spec.short)

    raw_markers = [ax_map.plot([], [], marker="x", markersize=9, markeredgewidth=2.2, color="#d8a600", linestyle="None", zorder=30)[0] for _ in range(2)]
    raw_circles = [Circle((0, 0), 1, fill=False, edgecolor="#d8a600", linewidth=1.4, linestyle=":", zorder=12) for _ in range(2)]
    ue_markers = [ax_map.plot([], [], marker="o", markersize=9, markerfacecolor="none", markeredgewidth=2, color=cfg["AC+CBF+UE"].color, linestyle="None", zorder=30)[0] for _ in range(2)]
    full_markers = [ax_map.plot([], [], marker="s", markersize=7, color=cfg["Full"].color, linestyle="None", zorder=31)[0] for _ in range(2)]
    full_circles = [Circle((0, 0), 1, fill=False, edgecolor=cfg["Full"].color, linewidth=1.8, linestyle="--", zorder=13) for _ in range(2)]
    for c in raw_circles + full_circles:
        ax_map.add_patch(c)

    ax_h.axhline(0, color="black", linewidth=1.6)
    ax_h.fill_between([0, display_time_s], -5, 0, color="#eeeeee", zorder=-3)
    ax_h.set_xlim(0, display_time_s)
    ax_h.set_ylim(-5, 30)
    ax_h.set_ylabel("h(t) (cm)")
    ax_h.set_xlabel("time (s)")
    ax_h.set_title("Minimum true signed clearance h(t)", fontsize=13, fontweight="bold")
    ax_h.grid(alpha=0.25)

    raw_e_line, = ax_e.plot([], [], color="#d8a600", linewidth=2.0, label="shared raw / no-UE")
    ue_e_line, = ax_e.plot([], [], color=cfg["AC+CBF+UE"].color, linewidth=2.2, label="UE-only")
    full_e_line, = ax_e.plot([], [], color=cfg["Full"].color, linewidth=3.0, label="Full")
    ax_e.set_xlim(0, display_time_s)
    max_err = max(18.0, 100 * raw.raw_center_rmse.max() * 1.15)
    ax_e.set_ylim(0, max_err)
    ax_e.set_ylabel("RMSE (cm)")
    ax_e.set_xlabel("time (s)")
    ax_e.set_title("Obstacle-center belief RMSE", fontsize=13, fontweight="bold")
    ax_e.grid(alpha=0.25)

    cursor_h = ax_h.axvline(0, color="#333333", linewidth=1)
    cursor_e = ax_e.axvline(0, color="#333333", linewidth=1)
    fig.suptitle("V17.1 severe shared sensor-position uncertainty", fontsize=20, fontweight="bold", y=0.96)
    subtitle = fig.text(0.5, 0.905, "", ha="center", va="center", fontsize=11, color="#333333")

    method_handles = [Line2D([0], [0], color=s.color, linewidth=s.line_width, label=s.short) for s in specs]
    belief_handles = [
        Line2D([0], [0], marker="o", markerfacecolor="#ffdede", markeredgecolor="#b31b1b", linestyle="None", label="fixed true obstacle (evaluation only)"),
        Line2D([0], [0], marker="x", color="#d8a600", linestyle="None", label="shared raw = no-UE belief"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=cfg["AC+CBF+UE"].color, linestyle="None", label="UE-only belief"),
        Line2D([0], [0], marker="s", color=cfg["Full"].color, linestyle="None", label="Full replay-calibrated belief"),
    ]
    fig.legend(handles=method_handles, loc="lower left", bbox_to_anchor=(0.04, 0.07), ncol=6, frameon=False, fontsize=9)
    fig.legend(handles=belief_handles, loc="lower left", bbox_to_anchor=(0.04, 0.015), ncol=4, frameon=False, fontsize=9)
    fig.text(0.70, 0.045, "unsafe samples are × on robot paths", ha="left", fontsize=9, color="#7a3a13")

    def update(frame_index: int):
        step = int(frame_steps[frame_index])
        t = step * world.dt
        artists: List[object] = []
        for spec in specs:
            d = methods[spec.key]
            cur = d[d.step <= step]
            path_lines[spec.key].set_data(100 * cur.x, 100 * cur.y)
            if len(cur):
                last = cur.iloc[-1]
                robot_dots[spec.key].set_data([100 * last.x], [100 * last.y])
            unsafe = cur[cur.violation > 0]
            unsafe_lines[spec.key].set_data(100 * unsafe.x, 100 * unsafe.y)
            h_lines[spec.key].set_data(cur.time, 100 * cur.true_clearance)
            artists += [path_lines[spec.key], robot_dots[spec.key], unsafe_lines[spec.key], h_lines[spec.key]]
        rr = raw.iloc[min(step, len(raw)-1)]
        ur = ue.iloc[min(step, len(ue)-1)]
        fr = full.iloc[min(step, len(full)-1)]
        for i, obs in enumerate(world.uncertain_obstacles, start=1):
            rx, ry = 100 * rr[f"obs{i}_raw_x"], 100 * rr[f"obs{i}_raw_y"]
            raw_markers[i-1].set_data([rx], [ry])
            raw_circles[i-1].center = (rx, ry)
            raw_circles[i-1].radius = 100 * obs.radius
            ux, uy = 100 * ur[f"obs{i}_belief_x"], 100 * ur[f"obs{i}_belief_y"]
            ue_markers[i-1].set_data([ux], [uy])
            fx, fy = 100 * fr[f"obs{i}_belief_x"], 100 * fr[f"obs{i}_belief_y"]
            full_markers[i-1].set_data([fx], [fy])
            full_circles[i-1].center = (fx, fy)
            full_circles[i-1].radius = 100 * (obs.radius + world.robot_radius + 0.009 + 1.20 * fr[f"obs{i}_sigma"])
            artists += [raw_markers[i-1], raw_circles[i-1], ue_markers[i-1], full_markers[i-1], full_circles[i-1]]
        cur_raw = raw[raw.step <= step]
        cur_ue = ue[ue.step <= step]
        cur_full = full[full.step <= step]
        raw_e_line.set_data(cur_raw.time, 100 * cur_raw.raw_center_rmse)
        ue_e_line.set_data(cur_ue.time, 100 * cur_ue.belief_center_rmse)
        full_e_line.set_data(cur_full.time, 100 * cur_full.belief_center_rmse)
        cursor_h.set_xdata([t, t]); cursor_e.set_xdata([t, t])
        subtitle.set_text(f"Red truth is evaluator-only; yellow × is the same raw measurement for every method; t = {t:.2f}/{display_end_s:.2f} s")
        artists += [raw_e_line, ue_e_line, full_e_line, cursor_h, cursor_e, subtitle]
        return artists

    anim = FuncAnimation(fig, update, frames=len(frame_steps), interval=1000 / fps, blit=False)
    anim_dir = out_dir / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    mp4 = anim_dir / f"v17_1_shared_sensor_uncertainty_stress_seed{display_seed}.mp4"
    gif = anim_dir / f"v17_1_shared_sensor_uncertainty_stress_seed{display_seed}.gif"
    try:
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=100)
    except Exception as exc:
        (anim_dir / "mp4_generation_error.txt").write_text(str(exc), encoding="utf-8")
    anim.save(gif, writer=PillowWriter(fps=fps), dpi=62)
    preview_step = int(0.66 * (len(frame_steps) - 1))
    update(preview_step)
    fig.savefig(anim_dir / "v17_1_animation_preview.png", dpi=100, facecolor="white")
    plt.close(fig)


def write_runtime_environment(root: Path, out_dir: Path) -> None:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy_available": SCIPY_AVAILABLE,
        "scipy": scipy.__version__ if SCIPY_AVAILABLE else None,
    }
    (out_dir / "runtime_environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def write_protocol_and_readme(
    root: Path,
    out_dir: Path,
    epochs: int,
    seeds: int,
    moderate_test: float,
    stress_test: float,
    disturbance_stress: float,
) -> None:
    write_runtime_environment(root, out_dir)
    display_seed_path = out_dir / "display_seed.txt"
    display_seed = int(display_seed_path.read_text(encoding="utf-8")) if display_seed_path.exists() else min(4, seeds - 1)

    def sensor_profile(multiplier: float) -> dict:
        return {
            "multiplier": multiplier,
            "true_obstacles_move": False,
            "episode_registration_bias_amplitude_m": [0.010 * multiplier, 0.019 * multiplier],
            "colored_noise_innovation_std_m": 0.0032 * math.sqrt(max(multiplier, 0.05)),
            "white_noise_std_m": 0.0026 * math.sqrt(max(multiplier, 0.05)),
            "sinusoidal_drift_amplitude_m": {"x": 0.0090 * multiplier, "y": 0.0080 * multiplier},
            "jump_initial_amplitude_m": [0.050 * multiplier, 0.070 * multiplier],
            "nominal_jump_decay_samples_including_injection": [16, 22],
            "nominal_jump_decay_s": [16 * World.dt, 22 * World.dt],
            "jump_injections_may_overlap": True,
            "hold_onset_probability_when_inactive": min(0.012 * multiplier, 0.055),
            "hold_duration_steps": [2, 4],
            "reported_detector_sigma_m": 0.010,
        }

    manifest = {
        "benchmark_version": "V17.1",
        "description": "Static hidden truth with one common detector stream; severity changes only perception mismatch.",
        "nominal": sensor_profile(1.0),
        "moderate": sensor_profile(moderate_test),
        "additional_extreme": sensor_profile(stress_test),
        "robustness_sweep_multipliers": [0.8, 1.0, 1.3, 1.6, 2.0, 2.2, 2.6, 3.0, 3.5, 4.0, 4.5],
    }
    (out_dir / "sensor_severity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    protocol = {
        "benchmark_version": "V17.1",
        "epochs": epochs,
        "seeds": seeds,
        "training_sensor_curriculum": "stress = 1.0 + 0.06 * epoch",
        "nominal_sensor_stress": 1.0,
        "moderate_sensor_multiplier": moderate_test,
        "additional_extreme_sensor_multiplier": stress_test,
        "additional_extreme_is_preregistered_confirmatory_test": False,
        "disturbance_stress": disturbance_stress,
        "cbf_disturbance_design_allowance_m_per_s": 0.012 * disturbance_stress,
        "stochastic_disturbance_is_hard_bounded": False,
        "display_seed": display_seed,
        "evaluation_episode_ids": {"nominal": 1000, "moderate": 1001, "severe": 1001},
        "robustness_sweep_episode_id": 2000,
        "robustness_sweep_levels": manifest["robustness_sweep_multipliers"],
        "world": {
            "width_m": World.width,
            "height_m": World.height,
            "robot_radius_m": World.robot_radius,
            "dt_s": World.dt,
            "train_horizon_steps": World.train_horizon,
            "evaluation_horizon_steps": World.eval_horizon,
            "evaluation_transition_horizon_s": World.eval_horizon * World.dt,
            "last_pretransition_sample_time_s": (World.eval_horizon - 1) * World.dt,
            "max_speed_m_per_s": World.max_speed,
            "goal_tolerance_m": World.goal_tolerance,
            "goal_success_definition": "ever enters the 0.055 m goal tolerance during the 280-transition episode",
        },
        "tuning_provenance": {
            "formal_hyperparameter_search_performed": False,
            "validation_set_used_for_parameter_selection": False,
            "method_specific_search_performed": False,
            "description": "One common hand-fixed engineering configuration is shared across the exploratory ablations; the values are not claimed tuned optima.",
            "posthoc_weight_sensitivity_used_for_selection": False,
            "posthoc_weight_sensitivity_design": "Each AER weight is perturbed one at a time by -20% or +20%, then all four weights are renormalized; five seeds are rerun at multipliers 2.2 and 6.0.",
        },
        "dynamics_and_objective": {
            "plant_update": "p_next = clip_workspace(p + 0.055 * (u_exec + disturbance))",
            "cost": "0.036*||p-goal||^2 + 0.020*||u_exec||^2 + 0.012*||u_exec-u_nom||^2 + 0.036*||u_exec-u_nom|| + 5.2*I_observed_contact + max(0,0.010-known_clearance)^2",
            "contact_is_observable_bumper_event": True,
            "continuous_true_clearance_used_by_controller": False,
        },
        "random_seed_formulas": {
            "actor_initialization": "12345 + seed",
            "sensor": "17000003 + 100003*seed + 1009*episode_id",
            "disturbance": "19001007 + 200003*seed + 1013*episode_id",
            "exploration": "81021 + 1000*seed + 53*episode_id",
            "replay": "202401 + 101*seed",
        },
        "implementation_hyperparameters": {
            "actor": {
                "initial_waypoints_m": World().initial_waypoints.tolist(),
                "initial_waypoint_jitter_std_m": 0.006,
                "exploration_std_m_per_s_per_component": WaypointActor.EXPLORATION_STD,
                "waypoint_attraction_gain": 2.25,
                "terminal_goal_gain": 0.28,
                "waypoint_switch_current_distance_m": 0.118,
                "waypoint_switch_next_advantage_m": 0.030,
                "learning_rate": 0.0040,
                "td_policy_gain": WaypointActor.TD_POLICY_GAIN,
                "td_clip": [-2.2, 2.2],
                "normalized_exploration_component_clip": [-2.5, 2.5],
                "auxiliary_pressure_weights_safety_uncertainty_contact": [0.58, 0.27, 0.15],
                "waypoint_smoothing_current_previous_next": [0.74, 0.13, 0.13],
                "internal_waypoint_corridor_bounds_xmin_xmax_ymin_ymax_m": [
                    [0.24, 0.38, 2.31, 2.46], [0.53, 0.77, 2.10, 2.36],
                    [0.77, 1.07, 1.84, 2.14], [0.84, 1.13, 1.52, 1.86],
                    [0.88, 1.11, 1.16, 1.47], [0.97, 1.11, 0.73, 0.89],
                    [1.29, 1.60, 0.65, 0.77], [2.02, 2.31, 0.65, 0.78],
                    [2.16, 2.43, 1.00, 1.26], [1.70, 1.94, 1.33, 1.55],
                ],
            },
            "critic": {
                "type": "linear state-value critic",
                "features": ["1", "goal_dx", "goal_dy", "goal_dx^2", "goal_dy^2", "x/3.2", "y/3.2"],
                "discount": 0.985,
                "learning_rate": 0.016,
                "td_clip": [-2.2, 2.2],
                "update_normalizer": "1 + ||features||^2",
            },
            "nominal_potential_fields": {
                "known_rectangles": {"extra_inflation_m": 0.022, "influence_m": 0.18, "gain": 0.42},
                "workspace_walls": {"extra_margin_m": 0.025, "influence_m": 0.14, "gain": 0.34},
                "perceived_circles": {"extra_inflation_m": "0.010 + 0.35*sigma", "influence_m": 0.31, "gain": 0.31},
            },
            "estimator": {
                "initial_nominal_scale_m": 0.018,
                "initial_learned_gate_m": 0.050,
                "reset_scale_floor_m": 0.014,
                "default_innovation_gate_m": 0.060,
                "base_gain_ue_full": [0.24, 0.18],
                "gain_denominator_floor_m": 0.035,
                "observable_jump_threshold_non_ue_ue": [0.60, 0.72],
                "variance_ema_old_new": [0.93, 0.07],
                "posterior_sigma_rule_m": "clip(0.008 + 1.35*sqrt(var) + outlier_add, 0.014, 0.075)",
                "outlier_add_ue_full_m": [0.012, 0.008],
                "episode_reset_anchor_raw_blend": [0.78, 0.22],
                "online_anchor_pull": "0.018 + 0.10*I_outlier",
                "calibration_minimum_batch": 8,
                "geometric_median_max_iterations": 50,
                "geometric_median_tolerance": 1e-8,
                "calibration_lower_distance_quantile": 0.60,
                "nominal_multiplier_and_clip_m": [1.45, 0.009, 0.030],
                "gate_multiplier_and_clip_m": [2.7, 0.030, 0.058],
                "anchor_ema_old_new": [0.86, 0.14],
                "nominal_and_gate_ema_old_new": [0.88, 0.12],
            },
            "cbf": {
                "known_extra_inflation_m": 0.018,
                "uncertain_robust_inflation_m": "0.009 + 1.20*sigma",
                "candidate_ranges_known_uncertain_walls_m": [0.30, 0.40, 0.26],
                "projection_active_below_h_m": 0.20,
                "class_k_gain": 2.9,
                "rhs_robust_coefficient": 0.18,
                "outer_alternating_cycles": 4,
                "all_barrier_passes_per_cycle": 1,
                "known_barrier_passes_before_and_after_speed_clip": [2, 2],
                "discrete_guard_base_m": 0.0015,
                "discrete_guard_max_iterations": 10,
                "final_known_barrier_passes": 3,
            },
            "replay": {
                "capacity": 360,
                "batch_size": 44,
                "batches_per_training_episode": 8,
                "sample_without_replacement": True,
                "priority_floor": 1e-5,
                "aer_full_weights_td_safety_uncertainty_novelty": list(ReplayBuffer.AER_WEIGHTS),
                "sampling_exponent_aer_full": 0.78,
                "sampling_exponent_per": 0.68,
                "uniform_replacement": "reservoir sampling",
                "prioritized_random_candidate_probability": 0.16,
                "prioritized_fallback_acceptance_probability": 0.035,
                "prioritized_default_candidate": "current minimum-priority entry",
                "novelty_grid_m": 0.18,
                "boundary_event_thresholds": {"estimated_h_m": 0.115, "correction_m_per_s": 0.020},
                "uncertainty_event_score_threshold": 0.48,
                "td_priority_refresh_scope": "sampled entries after each critic update",
            },
            "sensor": {
                "colored_ar_coefficient": 0.82,
                "colored_innovation_std_m": "0.0032*sqrt(perception_severity)",
                "white_jitter_std_m": "0.0026*sqrt(perception_severity)",
                "registration_bias_range_m": "[0.010,0.019]*perception_severity",
                "sinusoidal_amplitudes_m": "[0.009,0.008]*perception_severity",
                "jump_range_m": "[0.050,0.070]*perception_severity",
                "jump_decay_factor": 0.955,
                "hold_onset_probability": "min(0.012*perception_severity,0.055)",
                "hold_duration_steps": [2, 4],
                "reported_sigma_m": 0.010,
            },
            "disturbance": {
                "ar_coefficient": 0.90,
                "gaussian_innovation_std_m_per_s_at_multiplier_1": 0.0018,
                "sinusoidal_amplitudes_m_per_s": [0.0040, 0.0035],
                "hard_bounded": False,
            },
        },
        "fairness": {
            "same_map_and_hidden_truth": True,
            "same_raw_sensor_arrays_per_seed_episode_and_severity": True,
            "same_validity_bits": True,
            "same_observable_jumps": True,
            "same_exploration_draws": True,
            "same_plant_disturbance": True,
            "same_cost_horizon_and_seeds": True,
            "same_replay_rng_base_seed": True,
            "method_specific_sensor_corruption": False,
        },
        "information_flow": {
            "true_uncertain_centers_controller_visible": False,
            "true_clearance_controller_visible": False,
            "true_centers_stored_in_replay": False,
            "critic_uses_realized_post_filter_transition": True,
            "actor_uses_critic_td_residual": True,
            "actor_update": "replay-weighted clipped cost-TD residual times normalized common Gaussian exploration draw",
            "actor_update_scope": "biased off-policy score-function surrogate; no behavior-policy likelihood ratio or importance correction",
            "actor_td_policy_gain": WaypointActor.TD_POLICY_GAIN,
            "actor_exploration_std_m_per_s": WaypointActor.EXPLORATION_STD,
            "exploration_vector_stored_for_replayed_actor_update": True,
            "actor_waypoint_index_stored_for_replayed_actor_update": True,
            "behavior_policy_likelihood_stored": False,
            "actor_auxiliary_update": "controller-visible estimated-barrier gradient",
            "stress_scales_only_perception": True,
            "full_anchor_uses_true_center_labels": False,
            "moderate_and_severe_share_episode_seed": True,
            "robustness_sweep_uses_fixed_episode_seed_across_levels": True,
            "stress_6_independently_preregistered": False,
        },
        "priority_weights": {
            "score_order": ["TD", "safety", "uncertainty", "novelty"],
            "uniform_and_UE_only": None,
            "PER": [1.0, 0.0, 0.0, 0.0],
            "AER_and_Full": list(ReplayBuffer.AER_WEIGHTS),
            "sampling_exponent_PER": 0.68,
            "sampling_exponent_AER_and_Full": 0.78,
            "sample_without_replacement": True,
            "priority_floor": 1e-5,
        },
    }
    (out_dir / "experimental_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    def result_snapshot(filename: str) -> str:
        df = pd.read_csv(out_dir / filename)
        summary = df.groupby("method", as_index=False).agg(
            violation_count=("violation_count", "mean"),
            violating_seeds=("violation_count", lambda x: int((x > 0).sum())),
            min_clearance_cm=("min_clearance_cm", "mean"),
            belief_center_rmse_cm=("belief_center_rmse_cm", "mean"),
            goal_error_cm=("goal_error_cm", "mean"),
            reached_rate=("reached", "mean"),
            cost=("cost", "mean"),
        )
        order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
        summary["_order"] = summary.method.map(order_map)
        summary = summary.sort_values("_order").drop(columns="_order")
        lines = [
            "| Method | Mean violations | Seeds with violation | Mean minimum clearance (cm) | Belief RMSE (cm) | Goal error (cm) | Goal success | Mean cost |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.method} | {row.violation_count:.2f} | {row.violating_seeds}/{seeds} | "
                f"{row.min_clearance_cm:.2f} | {row.belief_center_rmse_cm:.2f} | "
                f"{row.goal_error_cm:.2f} | {100 * row.reached_rate:.0f}% | {row.cost:.2f} |"
            )
        return "\n".join(lines)

    moderate_snapshot = result_snapshot("moderate_eval_metrics.csv")
    severe_snapshot = result_snapshot("stress_eval_metrics.csv")
    severe_profile = sensor_profile(stress_test)

    readme = f"""# COMPELECENG V17.1 — severe shared sensor-position uncertainty benchmark

## Scientific purpose

The two circular obstacles are **physically static**, while their true centers are hidden from every controller. Every method receives exactly the same raw detector stream. V17.1 strengthens only the perception corruption so that the benchmark exposes the difference between raw-center CBF protection, online filtering, adaptive replay, and their integration.

This package does **not** weaken a baseline, move the true obstacles, give Full extra labels, or select a different sensor stream per method. The moderate post-training evaluation at multiplier `{moderate_test:g}` is retained alongside the additional exploratory extreme rerun at `{stress_test:g}`, and the robustness sweep is also saved. The software history records that `{stress_test:g}` was added after an earlier 4.0 run; it is not presented as a preregistered confirmatory test. A baseline is allowed to remain safe; failures are not an acceptance requirement.

## What is shown

- red filled circle: fixed true physical obstacle, evaluator-only;
- dark-red outer circle: true robot-collision boundary;
- yellow `×`: common noisy/held/jumping detector center;
- green marker: UE-only filtered belief;
- blue marker and dashed envelope: Full replay-calibrated belief and robust safety envelope;
- path `×`: a physical collision evaluated against hidden truth.

The figures and animation use the same trajectory CSV, so their geometry and safety curves cannot silently disagree.

## Severe perception profile

At `--stress-test {stress_test:g}`, the common detector has:

- episode registration-bias amplitude: {100 * severe_profile['episode_registration_bias_amplitude_m'][0]:.1f}–{100 * severe_profile['episode_registration_bias_amplitude_m'][1]:.1f} cm;
- smooth drift amplitudes: up to {100 * severe_profile['sinusoidal_drift_amplitude_m']['x']:.1f} cm in x and {100 * severe_profile['sinusoidal_drift_amplitude_m']['y']:.1f} cm in y;
- jump-outlier initial amplitude: {100 * severe_profile['jump_initial_amplitude_m'][0]:.0f}–{100 * severe_profile['jump_initial_amplitude_m'][1]:.0f} cm with a nominal {severe_profile['nominal_jump_decay_s'][0]:.2f}–{severe_profile['nominal_jump_decay_s'][1]:.2f} s decay per injection; scheduled injections can overlap;
- intermittent 2–4-frame holds, whose onset probability while inactive is {100 * severe_profile['hold_onset_probability_when_inactive']:.1f}% per step;
- detector-reported standard deviation fixed at 1.0 cm, deliberately representing an overconfident detector under distribution shift.

The true obstacle centers never move. `--disturbance-stress` independently scales the small common Gaussian AR(1)-plus-sinusoid plant disturbance. The CBF uses 0.012 m/s as a design allowance at disturbance multiplier 1.0; the stochastic generator is not hard bounded, so this allowance is not a safety certificate.

## Information available to each method

- `AC`, `AC+CBF`, `AC+CBF+PER`, and `AC+CBF+AER` use the common raw/held center;
- `AC+CBF+UE` uses a robust online constant-position filter;
- `Full` uses the same filter plus a static-center anchor learned self-supervised from adaptively replayed raw measurements.

No true center or true clearance enters the actor, CBF, critic, replay priority, estimator, or replay buffer. See `control_information_contract.json`, `sensor_fairness_audit.csv`, and `oracle_invariance_audit.csv`.

## Actor--critic update

The linear state-value critic is fitted from the realized cost and next state produced by the executed post-filter action. The waypoint actor uses a replay-weighted cost-advantage score-function update: the critic's clipped TD residual multiplies the normalized common Gaussian exploration draw, with the sign chosen to reduce cost. A controller-visible estimated-barrier gradient supplies an auxiliary safety-shaping term. Thus the critic directly changes the policy; it is not used only for replay priority. However, replayed transitions do not carry behavior-policy likelihoods and no importance ratios are applied, so this clipped, safety-shaped update is a biased off-policy surrogate rather than an unbiased on-policy policy-gradient estimator. The benchmark does not isolate a performance gain due to the value baseline alone. The common nominal policy also contains disclosed known-map and perceived-obstacle potential-field terms.

## Replay scores and priorities

Every stored transition uses controller-visible quantities only. The normalized scores are:

```text
TD score          = min(1, |δ_t| / 2.5)
safety score      = min(1, exp(-max(h_hat_t,0)/0.075)
                           + 0.34 I_CBF + 0.30 min(1, ||u-u_a||/0.11)
                           + 0.75 I_observed-contact)
uncertainty score = min(1, 0.58 mean(innovation_i / max(2.5 sigma_i,0.030))
                           + 0.30 mean(observable-jump_i)
                           + 0.28 (1-mean(valid_i)))
novelty score     = clipped state-cell novelty in [0,1]
```

Uniform replay (including `AC+CBF+UE`) samples uniformly. `AC+CBF+PER` uses TD priority only. `AC+CBF+AER` and `Full` use

```text
p_t = 0.18 TD + 0.39 safety + 0.30 uncertainty + 0.13 novelty + 1e-5.
```

Prioritized mini-batches are sampled **without replacement**. At each sequential draw, `AC+CBF+PER` uses probability proportional to `p_t**0.68`, while `AC+CBF+AER` and `Full` use probability proportional to `p_t**0.78`; the denominator is recomputed over entries not yet selected for that batch. Uniform rows sample without replacement at equal probability. Uniform insertion uses reservoir replacement. Prioritized replacement chooses a random candidate with probability `0.16` and otherwise the current minimum-priority entry; the new entry is accepted when its priority is no smaller, or with fallback probability `0.035`. Only sampled entries have their TD-dependent priority refreshed after each critic update.

`demo_results/experimental_protocol.json` records the complete plant, cost, actor, critic, waypoint/corridor, potential-field, estimator, CBF, sensor, disturbance, replay, seed, and selection-provenance constants. No formal validation-set or method-specific hyperparameter search is claimed.

The finite-buffer retention and sampling fractions are saved in `replay_finite_buffer_audit.csv` and `paper_ready_replay_audit_table.md`.

The separate `run_posthoc_aer_weight_sensitivity.py` diagnostic perturbs each nominal AER weight by -20% and +20% one at a time, renormalizes the four weights, and reruns Full at the moderate and exploratory extreme tiers. It is descriptive and was not used for tuning or headline selection. Its aggregate and seed-level CSV/Markdown outputs are stored under `demo_results/posthoc_aer_weight_sensitivity_*`.

## Reproduce the included run

```bash
python run_reproducible_demo.py --out demo_results --epochs {epochs} --seeds {seeds} --moderate-test {moderate_test} --stress-test {stress_test} --disturbance-stress {disturbance_stress} --display-seed {display_seed} --animate --clean
```

Windows Command Prompt:

```bat
run_demo.bat
```

## Included moderate result

{moderate_snapshot}

## Included severe result

{severe_snapshot}

Interpret the severe result jointly. The intended evidence is not “every ablation must collide.” It is that Full preserves safety, reaches the target, maintains positive true clearance, reduces hidden-center estimation error using only shared measurements, and remains competitive in cost. UE-only may remain collision-free but can become conservative or fail to finish; that is a legitimate ablation outcome rather than something to hide.

## Main outputs

- `demo_results/moderate_eval_metrics.csv`
- `demo_results/stress_eval_metrics.csv`
- `demo_results/evaluation_trajectories_seed0.csv`
- `demo_results/evaluation_trajectories_display_seed{display_seed}.csv`
- `demo_results/shared_sensor_stream_severe_seed0.csv`
- `demo_results/sensor_severity_manifest.json`
- `demo_results/sensor_fairness_audit.csv`
- `demo_results/oracle_invariance_audit.csv`
- `demo_results/replay_finite_buffer_audit.csv`
- `demo_results/robustness_sweep.csv`
- `demo_results/robustness_sweep_collision_source_audit.csv`
- `demo_results/paper_ready_evaluation_table.md/.tex`
- `demo_results/paper_ready_moderate_evaluation_table.md/.tex`
- `demo_results/figures/fig01...fig13`
- `demo_results/animations/v17_1_shared_sensor_uncertainty_stress_seed{display_seed}.mp4`
- `demo_results/animations/v17_1_shared_sensor_uncertainty_stress_seed{display_seed}.gif`
- `demo_results/simulation_acceptance_audit_v17_1.md`

Aggregate tables use all seeds. Display seed {display_seed} is visualization-only and is explicitly recorded in the protocol.

## Install and test

```bash
python -m pip install -r requirements.txt
python -m unittest -v tests/test_shared_sensor_and_no_oracle.py
```

## Postprocess an existing completed run

```bash
python postprocess_results.py --out demo_results --epochs {epochs} --seeds {seeds} --moderate-test {moderate_test} --stress-test {stress_test} --disturbance-stress {disturbance_stress} --animate
```
"""
    # Keep run-specific snapshots with their output. Do not overwrite the
    # repository's public README when users run a smaller smoke configuration.
    (out_dir / "RUN_README.md").write_text(readme, encoding="utf-8")

    correction = f"""# V17.1 design and severity correction

V17.1 preserves the information-flow correction: hidden static truth is separated from one common noisy detector stream and method-specific post-processing. The software history records an earlier severe multiplier of 4.0 and the later addition of the reported {stress_test:g} stress case while retaining the moderate {moderate_test:g} case. The {stress_test:g} case was not independently preregistered; it is reported as a designed stress benchmark, not as an untouched confirmatory test.

## Fair comparison

All methods share the map, hidden truth, start and goal, actor initialization, exploration, raw sensor arrays, validity bits, detector jumps, disturbance, cost, horizon, and evaluation seeds. No method-specific sensor corruption is permitted.

## Why stronger uncertainty is legitimate

The severe setting models a detector under registration shift, correlated error, temporary holds, and large but visible outliers. It is an out-of-distribution perception test, not an attempt to sabotage baselines. The same episode seed is evaluated at moderate and severe multipliers, and the full robustness sweep is reported.

## Why Full may estimate better

Full receives no ground-truth center labels. Adaptive replay retains boundary and innovation events, and the estimator forms a robust geometric-median anchor from replayed raw measurements. The claim must be supported jointly by common-stream fairness, lower belief RMSE, safety, target completion, clearance, cost, and replay audits.
"""
    (out_dir / "DESIGN_CORRECTION.md").write_text(correction, encoding="utf-8")

def write_acceptance_audit(out_dir: Path) -> None:
    stress = pd.read_csv(out_dir / "stress_eval_metrics.csv")
    moderate_path = out_dir / "moderate_eval_metrics.csv"
    moderate = pd.read_csv(moderate_path) if moderate_path.exists() else None
    fairness = pd.read_csv(out_dir / "sensor_fairness_audit.csv")
    oracle = pd.read_csv(out_dir / "oracle_invariance_audit.csv")

    means = stress.groupby("method", as_index=False).agg(
        violation_count=("violation_count", "mean"),
        violating_seeds=("violation_count", lambda x: int((x > 0).sum())),
        min_clearance_cm=("min_clearance_cm", "mean"),
        worst_seed_clearance_cm=("min_clearance_cm", "min"),
        min_uncertain_true_clearance_cm=("min_uncertain_true_clearance_cm", "mean"),
        cost=("cost", "mean"),
        belief_center_rmse_cm=("belief_center_rmse_cm", "mean"),
        goal_error_cm=("goal_error_cm", "mean"),
        reached_rate=("reached", "mean"),
        known_static_violation_count=("known_static_violation_count", "mean"),
    )
    full = stress[stress.method == "Full"]
    full_safe = bool(
        full.violation_count.eq(0).all()
        and (full.min_clearance_cm > 0).all()
        and full.reached.eq(1).all()
    )
    known_ok = bool(stress.known_static_violation_count.eq(0).all())
    fairness_ok = bool(fairness.pass_identical_common_sensor.eq(1).all())
    oracle_ok = bool(oracle.pass_no_oracle_dependence.eq(1).all())
    rmse = means.set_index("method").belief_center_rmse_cm
    estimator_ok = bool(rmse["Full"] < rmse["AC+CBF+UE"] < rmse["AC"])

    non_full = stress[stress.method != "Full"]
    methods_with_any_collision = int(
        non_full.groupby("method").violation_count.sum().gt(0).sum()
    )
    separation_informative = methods_with_any_collision >= 3

    required = [
        out_dir / "figures" / "fig01_full_trajectory_comparison.png",
        out_dir / "figures" / "fig02_full_method_trajectory.png",
        out_dir / "figures" / "fig11_obstacle_belief_rmse_over_time.png",
    ]
    animation_mp4 = sorted(
        (out_dir / "animations").glob("v17_1_shared_sensor_uncertainty_stress_seed*.mp4")
    )
    animation_gif = sorted(
        (out_dir / "animations").glob("v17_1_shared_sensor_uncertainty_stress_seed*.gif")
    )
    files_ok = (
        all(p.exists() and p.stat().st_size > 0 for p in required)
        and any(p.stat().st_size > 0 for p in animation_mp4)
        and any(p.stat().st_size > 0 for p in animation_gif)
    )
    # Baseline failure is a diagnostic, not a validity requirement.
    overall = full_safe and known_ok and fairness_ok and oracle_ok and estimator_ok and files_ok
    lines = [
        "# V17.1 release regression audit",
        "",
        f"**Regression status: {'PASS' if overall else 'FAIL'}**",
        "",
        "This outcome-dependent audit checks reproduction of the released run; it is not independent validation or a confirmatory acceptance criterion.",
        "",
        f"- Shared raw sensor stream verified across methods and evaluation tiers: {'PASS' if fairness_ok else 'FAIL'}",
        f"- Hidden true-center oracle invariance: {'PASS' if oracle_ok else 'FAIL'}",
        f"- Full zero collisions, positive clearance, and target completion in every severe seed: {'PASS' if full_safe else 'FAIL'}",
        f"- Zero known-static-wall collisions across all methods: {'PASS' if known_ok else 'FAIL'}",
        f"- Belief RMSE ordering Full < UE-only < raw/no-UE: {'PASS' if estimator_ok else 'FAIL'}",
        f"- Required figures and MP4/GIF exist: {'PASS' if files_ok else 'FAIL'}",
        "",
        "## Benchmark informativeness diagnostic",
        "",
        f"- Non-Full methods with at least one severe-test collision: {methods_with_any_collision}/5.",
        f"- At least three non-Full methods exhibit a collision: {'YES' if separation_informative else 'NO'}.",
        "- This diagnostic is deliberately not an acceptance criterion; a valid ablation is allowed to remain safe.",
        "",
        "## Severe-test means",
        "",
        means.to_markdown(index=False),
    ]
    if moderate is not None:
        moderate_means = moderate.groupby("method", as_index=False).agg(
            violation_count=("violation_count", "mean"),
            min_clearance_cm=("min_clearance_cm", "mean"),
            belief_center_rmse_cm=("belief_center_rmse_cm", "mean"),
            goal_error_cm=("goal_error_cm", "mean"),
            reached_rate=("reached", "mean"),
            cost=("cost", "mean"),
        )
        lines += ["", "## Moderate-test means", "", moderate_means.to_markdown(index=False)]
    lines += [
        "",
        "## Information-flow audit",
        "",
        "- True uncertain-obstacle centers are used only for sensor generation, physical evaluation, and figures.",
        "- The replay buffer stores a sanitized controller-visible transition without true centers, true clearances, hidden fault labels, or evaluator errors.",
        "- Actor and CBF receive method beliefs, not evaluator truth.",
        "- Replay safety priority uses estimated barrier slack, correction, CBF activation, and observed contact.",
        "- Replay uncertainty priority uses innovation, visible frame-to-frame measurement change, and validity bits.",
        "- The state-value critic uses realized cost and next state produced by the executed safe action; its TD residual directly drives a biased replay-weighted score-function actor surrogate without behavior-policy importance correction.",
        "- Full replay calibration uses only raw measurement samples and a geometric median.",
        "- Moderate and additional extreme tests use the same episode seed; severity scales error amplitudes and detector hold-onset probability.",
        "- Robustness-sweep levels use one fixed episode seed, reducing the severity/random-realization confound.",
    ]
    (out_dir / "simulation_acceptance_audit_v17_1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

def write_manifest(root: Path) -> Path:
    """Write SHA-256 entries for every packaged file except the manifest itself."""
    excluded = {"__pycache__", ".pytest_cache", ".git"}
    manifest_path = root / "MANIFEST.sha256"
    lines: List[str] = []
    for p in sorted(root.rglob("*")):
        if (
            not p.is_file()
            or p == manifest_path
            or any(part in excluded for part in p.parts)
            or p.suffix in {".pyc", ".zip"}
        ):
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {p.relative_to(root).as_posix()}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def package_zip(root: Path, output: Optional[Path] = None) -> Path:
    write_manifest(root)
    if output is None:
        output = root.parent / f"{root.name}.zip"
    output = output.resolve()
    if output.exists():
        output.unlink()
    excluded = {"__pycache__", ".pytest_cache", ".git"}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as zf:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or any(part in excluded for part in p.parts):
                continue
            if p.suffix in {".pyc"}:
                continue
            zf.write(p, p.relative_to(root.parent))
    return output


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="V17.1 severe shared-sensor uncertainty benchmark")
    ap.add_argument("--out", default="demo_results")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--moderate-test", type=float, default=2.2, help="secondary moderate perception multiplier")
    ap.add_argument("--stress-test", type=float, default=6.0, help="additional exploratory extreme perception multiplier")
    ap.add_argument("--disturbance-stress", type=float, default=1.0)
    ap.add_argument("--display-seed", type=int, default=4)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--zip", action="store_true")
    return ap.parse_args()
