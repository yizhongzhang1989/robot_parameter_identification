"""Offline measured-pose selection and callback-only transit screening."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class NoExecutedPoses(RuntimeError):
    """The observation table has no readable poses to revisit."""


def executed_poses(folder: Path, names) -> np.ndarray:
    """Read ordered joint positions and deduplicate at 0.1-degree precision."""
    rows = []
    with (Path(folder) / "observations.csv").open(
            newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append([float(row[f"{name}.position_deg"]) for name in names])
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise NoExecutedPoses("the run has no poses to revisit")
    return np.unique(np.round(np.array(rows), 1), axis=0)


def spread_then_order(poses, load, count: int) -> np.ndarray:
    """Pick posture-diverse poses, then order them lightest load first."""
    poses = np.asarray(poses, dtype=float)
    load = np.asarray(load, dtype=float)
    if count <= 0 or poses.shape[0] == 0:
        return np.zeros(0, dtype=int)
    count = min(int(count), poses.shape[0])
    chosen = [int(np.argmin(load))]
    while len(chosen) < count:
        gap = np.min([np.linalg.norm(poses - poses[index], axis=1)
                      for index in chosen], axis=0)
        gap[chosen] = -1.0
        chosen.append(int(np.argmax(gap)))
    return np.asarray(sorted(chosen, key=lambda index: load[index]), dtype=int)


def random_then_order(poses, load, count: int, seed: int) -> np.ndarray:
    """Pick distinct poses at random, then order them lightest load first."""
    poses = np.asarray(poses, dtype=float)
    load = np.asarray(load, dtype=float)
    if count <= 0 or poses.shape[0] == 0:
        return np.zeros(0, dtype=int)
    count = min(int(count), poses.shape[0])
    chosen = np.random.default_rng(seed).choice(
        poses.shape[0], size=count, replace=False)
    return np.asarray(sorted(chosen.tolist(), key=lambda index: load[index]),
                      dtype=int)


def admissible(gravity_a, link_z, envelope_a, minimum_z: float) -> np.ndarray:
    """Require every current within its envelope and the minimum link height."""
    gravity_a = np.abs(np.asarray(gravity_a, dtype=float))
    return (gravity_a <= np.asarray(envelope_a, dtype=float)).all(axis=1) & (
        np.asarray(link_z, dtype=float) >= float(minimum_z))


def transit_clear(start, goal, contacts, steps: int = 400) -> bool:
    """Screen the straight joint-space transit, including both endpoints."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    return not any(contacts(start + (goal - start) * alpha)
                   for alpha in np.linspace(0.0, 1.0, steps))
