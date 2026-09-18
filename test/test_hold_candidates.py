"""Pure candidate parity against the pre-extraction legacy algorithms."""

import ast
import csv
import io
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robot_parameter_identification import hold_candidates


def _legacy_executed_poses(folder, names):
    rows = []
    with (Path(folder) / "observations.csv").open(
            newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append([float(row[f"{name}.position_deg"]) for name in names])
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise RuntimeError("the run has no poses to revisit")
    return np.unique(np.round(np.array(rows), 1), axis=0)


def _legacy_spread_then_order(poses, load, count):
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


def _legacy_random_then_order(poses, load, count, seed):
    poses = np.asarray(poses, dtype=float)
    load = np.asarray(load, dtype=float)
    if count <= 0 or poses.shape[0] == 0:
        return np.zeros(0, dtype=int)
    count = min(int(count), poses.shape[0])
    chosen = np.random.default_rng(seed).choice(
        poses.shape[0], size=count, replace=False)
    return np.asarray(sorted(chosen.tolist(), key=lambda index: load[index]), dtype=int)


def _legacy_admissible(gravity_a, link_z, envelope_a, minimum_z):
    gravity_a = np.abs(np.asarray(gravity_a, dtype=float))
    return (gravity_a <= np.asarray(envelope_a, dtype=float)).all(axis=1) & (
        np.asarray(link_z, dtype=float) >= float(minimum_z))


def _legacy_transit_clear(start, goal, contacts, steps=400):
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    return not any(contacts(start + (goal - start) * alpha)
                   for alpha in np.linspace(0.0, 1.0, steps))


class ExecutedPosesTest(unittest.TestCase):
    def compare(self, table, names):
        with patch.object(Path, "open", side_effect=lambda *args, **kwargs: io.StringIO(table)):
            actual = hold_candidates.executed_poses(Path("/offline"), names)
            expected = _legacy_executed_poses(Path("/offline"), names)
        np.testing.assert_array_equal(actual, expected)
        return actual

    def test_ordered_columns_round_before_deduplication_and_sort(self):
        table = (
            "joint_b.position_deg,joint_a.position_deg,unused\n"
            "2.24,1.24,first\n2.21,1.21,duplicate\n-4.26,3.26,last\n"
            "bad,4,malformed\n1\n"
        )
        actual = self.compare(table, ["joint_a", "joint_b"])
        np.testing.assert_array_equal(actual, [[1.2, 2.2], [3.3, -4.3]])

    def test_rounding_matches_numpy_half_even_behavior(self):
        actual = self.compare(
            "joint.position_deg\n0.05\n0.15\n0.25\n-0.05\n-0.15\n-0.25\n",
            ["joint"])
        np.testing.assert_array_equal(actual, [[-0.2], [0.0], [0.2]])

    def test_joint_count_and_instance_names_are_not_hardware_restricted(self):
        for count in (1, 3, 7, 9):
            with self.subTest(count=count):
                names = [f"cell_axis_{index}" for index in range(count)]
                table = io.StringIO()
                writer = csv.writer(table)
                writer.writerow([f"{name}.position_deg" for name in names])
                writer.writerow(np.arange(count) + 0.04)
                writer.writerow(np.arange(count) + 0.01)
                actual = self.compare(table.getvalue(), names[::-1])
                np.testing.assert_array_equal(actual, [np.arange(count)[::-1]])

    def test_nonfinite_rows_remain_for_the_planners_finite_guard(self):
        actual = self.compare("joint.position_deg\n1\nnan\ninf\n-inf\n", ["joint"])
        self.assertEqual(actual.shape, (4, 1))
        self.assertEqual(np.count_nonzero(np.isfinite(actual)), 1)

    def test_empty_or_unreadable_selected_columns_raise_the_same_error(self):
        for table in ("", "other.position_deg\n1\n", "joint.position_deg\ninvalid\n\n"):
            with self.subTest(table=table):
                with patch.object(Path, "open", side_effect=lambda *args, **kwargs: io.StringIO(table)):
                    with self.assertRaisesRegex(hold_candidates.NoExecutedPoses, "no poses"):
                        hold_candidates.executed_poses(Path("/offline"), ["joint"])
                    with self.assertRaisesRegex(RuntimeError, "no poses"):
                        _legacy_executed_poses(Path("/offline"), ["joint"])

    def test_missing_observations_propagates_without_a_fallback_source(self):
        with patch.object(Path, "open", side_effect=FileNotFoundError("observations.csv")):
            for getter in (hold_candidates.executed_poses, _legacy_executed_poses):
                with self.subTest(getter=getter.__name__), self.assertRaises(FileNotFoundError):
                    getter(Path("/offline"), ["joint"])


class CandidateSelectionParityTest(unittest.TestCase):
    def test_seeded_random_selection_matches_legacy_for_counts_ties_and_dimensions(self):
        for dimensions in (1, 3, 7, 9):
            poses = np.random.default_rng(17).normal(size=(19, dimensions))
            load = np.arange(19) % 4
            for count in (-1, 0, 1, 7, 30):
                for seed in (0, 11, 2**63 - 1):
                    with self.subTest(dimensions=dimensions, count=count, seed=seed):
                        np.testing.assert_array_equal(
                            hold_candidates.random_then_order(poses, load, count, seed),
                            _legacy_random_then_order(poses, load, count, seed))

    def test_spread_selection_matches_legacy_for_counts_ties_and_dimensions(self):
        for dimensions in (1, 3, 7, 9):
            for poses in (np.zeros((19, dimensions)),
                          np.random.default_rng(17).normal(size=(19, dimensions))):
                load = np.arange(19) % 4
                for count in (-1, 0, 1, 7, 30):
                    with self.subTest(dimensions=dimensions, count=count):
                        np.testing.assert_array_equal(
                            hold_candidates.spread_then_order(poses, load, count),
                            _legacy_spread_then_order(poses, load, count))

    def test_empty_pools_produce_empty_integer_indices(self):
        for dimensions in (1, 7):
            poses, load = np.empty((0, dimensions)), np.empty(0)
            for select in (hold_candidates.spread_then_order, _legacy_spread_then_order,
                           hold_candidates.random_then_order, _legacy_random_then_order):
                options = {"seed": 11} if "random" in select.__name__ else {}
                indices = select(poses, load, 4, **options)
                self.assertEqual(indices.shape, (0,))
                self.assertTrue(np.issubdtype(indices.dtype, np.integer))

    def test_selection_never_mutates_inputs_or_global_random_state(self):
        poses = np.arange(70, dtype=float).reshape(10, 7)
        load = np.arange(10, dtype=float)
        before_poses, before_load = poses.copy(), load.copy()
        random_state = np.random.get_state()
        hold_candidates.spread_then_order(poses, load, 5)
        hold_candidates.random_then_order(poses, load, 5, 11)
        np.testing.assert_array_equal(poses, before_poses)
        np.testing.assert_array_equal(load, before_load)
        after_state = np.random.get_state()
        self.assertEqual(random_state[0], after_state[0])
        np.testing.assert_array_equal(random_state[1], after_state[1])
        self.assertEqual(random_state[2:], after_state[2:])

    def test_admission_matches_legacy_at_current_height_and_nonfinite_boundaries(self):
        for dimensions in (1, 3, 7, 9):
            envelope = np.arange(1, dimensions + 1, dtype=float)
            gravity = np.vstack([envelope, -envelope, envelope + 0.001,
                                 np.zeros(dimensions), np.full(dimensions, np.nan),
                                 np.full(dimensions, np.inf)])
            height = np.asarray([0.5, 0.5, 0.5, 0.499, 0.5, 0.5])
            actual = hold_candidates.admissible(gravity, height, envelope, 0.5)
            np.testing.assert_array_equal(actual, _legacy_admissible(
                gravity, height, envelope, 0.5))
            np.testing.assert_array_equal(actual, [True, True, False, False, False, False])


class TransitParityTest(unittest.TestCase):
    def compare(self, dimensions, stop_at, steps=400):
        start = np.arange(dimensions, dtype=float)
        goal = start + 40.0
        sampled = []
        verdicts = []
        for transit in (hold_candidates.transit_clear, _legacy_transit_clear):
            visited = []

            def contacts(pose):
                visited.append(pose.copy())
                return len(visited) == stop_at

            verdicts.append(transit(start, goal, contacts, steps))
            sampled.append(visited)
        self.assertEqual(verdicts[0], verdicts[1])
        np.testing.assert_array_equal(sampled[0], sampled[1])
        np.testing.assert_array_equal(start, np.arange(dimensions))
        np.testing.assert_array_equal(goal, np.arange(dimensions) + 40.0)
        return verdicts[0], sampled[0]

    def test_clear_transit_matches_all_400_samples_including_endpoints(self):
        for dimensions in (1, 3, 7, 9):
            clear, samples = self.compare(dimensions, None)
            self.assertTrue(clear)
            self.assertEqual(len(samples), 400)
            np.testing.assert_array_equal(samples[0], np.arange(dimensions))
            np.testing.assert_array_equal(samples[-1], np.arange(dimensions) + 40.0)

    def test_blocked_start_midpoint_and_goal_short_circuit_at_the_same_sample(self):
        for stop_at in (1, 201, 400):
            clear, samples = self.compare(7, stop_at)
            self.assertFalse(clear)
            self.assertEqual(len(samples), stop_at)

    def test_custom_step_counts_preserve_the_legacy_callback_contract(self):
        for steps in (0, 1, 2, 17):
            clear, samples = self.compare(3, None, steps)
            self.assertTrue(clear)
            self.assertEqual(len(samples), steps)


class LegacyForwarderTest(unittest.TestCase):
    def forwarder(self):
        path = Path(__file__).resolve().parents[3] / "tools" / "identified_static_hold_campaign.py"
        if not path.is_file():
            self.skipTest("workspace legacy forwarder is not available")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        statements = [node for node in tree.body if (
            isinstance(node, ast.ImportFrom) and
            node.module == "robot_parameter_identification.hold_candidates") or (
            isinstance(node, ast.FunctionDef) and node.name == "_executed_poses")]
        namespace = {"Path": Path, "np": np,
                     "identified": SimpleNamespace(ModelUnavailable=RuntimeError)}
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)
        return namespace

    def test_legacy_utilities_are_the_shared_functions(self):
        namespace = self.forwarder()
        for name in ("admissible", "random_then_order", "spread_then_order", "transit_clear"):
            self.assertIs(namespace[name], getattr(hold_candidates, name))
        with patch.object(Path, "open", side_effect=lambda *args, **kwargs: io.StringIO(
                "joint.position_deg\n1.24\n1.21\n")):
            np.testing.assert_array_equal(namespace["_executed_poses"]("/offline", ["joint"]),
                                          [[1.2]])

    def test_legacy_empty_pool_keeps_its_exception_contract(self):
        namespace = self.forwarder()
        with patch.object(Path, "open", return_value=io.StringIO("joint.position_deg\n")):
            with self.assertRaisesRegex(RuntimeError, "no poses") as raised:
                namespace["_executed_poses"]("/offline", ["joint"])
        self.assertIs(type(raised.exception), RuntimeError)
        self.assertIsInstance(raised.exception.__cause__, hold_candidates.NoExecutedPoses)


class CandidateImportTest(unittest.TestCase):
    def test_import_needs_no_runtime_files_commands_network_or_legacy_modules(self):
        path = Path(hold_candidates.__file__)
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        before = sys.path[:]
        forbidden = dict.fromkeys((
            "identified_static_hold_campaign", "identified_zero_force_drag",
            "forward_current_controller_test", "rclpy", "xacro",
            "ament_index_python.packages"))
        namespace = {"__name__": "hold_candidates_import_probe", "__file__": str(path)}
        with patch.dict(sys.modules, forbidden), \
                patch("builtins.open", side_effect=AssertionError("no runtime file reads")), \
                patch.object(Path, "open", side_effect=AssertionError("no runtime file reads")), \
                patch.object(subprocess, "run", side_effect=AssertionError("no commands")), \
                patch.object(subprocess, "Popen", side_effect=AssertionError("no processes")), \
                patch.object(socket, "socket", side_effect=AssertionError("no sockets")), \
                patch.object(socket, "create_connection", side_effect=AssertionError("no network")):
            exec(code, namespace)
            for name in forbidden:
                self.assertIsNone(sys.modules[name])
        self.assertEqual(sys.path, before)
        for name in ("executed_poses", "spread_then_order", "random_then_order",
                     "admissible", "transit_clear"):
            self.assertTrue(callable(namespace[name]))


if __name__ == "__main__":
    unittest.main()
