"""Rigid-body plus friction identification in the current domain.

Robot dynamics are linear in the inertial parameters, so one constant parameter
vector explains every configuration: a joint that feels light near zero and
heavy with the shoulder out is the same model evaluated at two configurations,
not two calibrations. What the experiment must do is visit enough configurations
for those constant parameters to become observable.

Because this arm reports current rather than torque, each joint is regressed
separately and its parameters absorb that joint's unknown torque constant. The
result predicts current directly, which is what the impedance controller needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pinocchio as pin

from .model import DEFAULT_COMPONENTS, ModelComponents, extra_row

# Samples needed inside a candidate reversal before it is worth considering.
MINIMUM_TRANSITION_SAMPLES = 12
# Relative residual improvement a fitted width must show before it displaces
# the configured one. Hardware gains were eleven to sixty per cent; anything
# near the noise is the split between Coulomb and viscous drifting, not a
# better measurement of the reversal.
TRANSITION_GAIN = 0.05

PARAMETERS_PER_LINK = 10
FRICTION_COLUMNS = ("coulomb", "viscous", "offset")

# Physics fixes the sign of these: Coulomb friction opposes motion, damping
# dissipates, reflected inertia is a mass, and load cannot make friction
# smaller. `stribeck` carries F_static - F_coulomb, so >= 0 is exactly
# "static friction is at least dynamic friction".
NONNEGATIVE_COLUMNS = (
    "coulomb", "viscous", "stribeck", "actuator_inertia", "load_friction")
PHYSICAL_PASSES = 4


def urdf_from_xacro(xacro_path: str | Path) -> str:
    result = subprocess.run(
        ["xacro", str(xacro_path)], capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"xacro failed: {result.stderr.strip()[:200]}")
    return result.stdout


@dataclass
class ArmModel:
    """One arm of the workspace URDF, reduced to its own joints."""

    model: pin.Model
    data: pin.Data
    joint_names: list[str]

    @classmethod
    def from_urdf_text(cls, urdf_text: str, prefix: str) -> "ArmModel":
        return cls._reduced(
            urdf_text, lambda name: name.startswith(prefix), repr(prefix))

    @classmethod
    def from_profile(cls, urdf_text: str, profile) -> "ArmModel":
        """Keep exactly the joints the profile names, in the model's own order."""
        wanted = set(profile.joint_names)
        model = cls._reduced(
            urdf_text, lambda name: name in wanted, f"profile {profile.name}")
        missing = wanted.difference(model.joint_names)
        if missing:
            raise ValueError(
                f"URDF is missing joints named by profile {profile.name}: "
                f"{sorted(missing)}")
        return model

    @classmethod
    def _reduced(cls, urdf_text: str, keep_joint, described: str) -> "ArmModel":
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as handle:
            handle.write(urdf_text)
            path = Path(handle.name)
        try:
            full = pin.buildModelFromUrdf(str(path))
        finally:
            path.unlink(missing_ok=True)

        keep, lock = [], []
        for index in range(1, full.njoints):
            (keep if keep_joint(full.names[index]) else lock).append(index)
        if not keep:
            raise ValueError(f"no joints matched {described}")
        reduced = pin.buildReducedModel(full, lock, pin.neutral(full))
        names = [reduced.names[i] for i in range(1, reduced.njoints)]
        return cls(reduced, reduced.createData(), names)

    @property
    def joint_count(self) -> int:
        return self.model.nv

    @property
    def parameter_count(self) -> int:
        return PARAMETERS_PER_LINK * (self.model.njoints - 1)

    def limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.degrees(self.model.lowerPositionLimit),
            np.degrees(self.model.upperPositionLimit),
        )

    def torque_regressor(self, q_deg, v_deg_s=None, a_deg_s2=None) -> np.ndarray:
        """Rows are joints, columns are the ten inertial parameters per link."""
        q = np.radians(np.asarray(q_deg, dtype=float))
        v = np.radians(np.asarray(
            np.zeros(self.joint_count) if v_deg_s is None else v_deg_s, dtype=float))
        a = np.radians(np.asarray(
            np.zeros(self.joint_count) if a_deg_s2 is None else a_deg_s2, dtype=float))
        return np.array(
            pin.computeJointTorqueRegressor(self.model, self.data, q, v, a),
        )

    def static_regressor(self, q_deg) -> np.ndarray:
        """Joint gravity regressor: the torque regressor at rest.

        A static experiment cannot see the inertia tensor, so those columns come
        out identically zero and are dropped by the base-parameter reduction.
        """
        return self.torque_regressor(q_deg)

    def align_base(self, translation, rotation) -> None:
        """Match another engine's mounting so both see the same gravity direction."""
        placement = self.model.jointPlacements[1]
        placement.translation = np.asarray(translation, dtype=float)
        placement.rotation = np.asarray(rotation, dtype=float)
        self.data = self.model.createData()

    def inertial_parameters(self) -> np.ndarray:
        values = []
        for index in range(1, self.model.njoints):
            values.extend(self.model.inertias[index].toDynamicParameters())
        return np.asarray(values, dtype=float)

    def inverse_dynamics(self, q_deg, v_deg_s=None, a_deg_s2=None) -> np.ndarray:
        q = np.radians(np.asarray(q_deg, dtype=float))
        v = np.radians(np.asarray(
            np.zeros(self.joint_count) if v_deg_s is None else v_deg_s, dtype=float))
        a = np.radians(np.asarray(
            np.zeros(self.joint_count) if a_deg_s2 is None else a_deg_s2, dtype=float))
        return np.array(pin.rnea(self.model, self.data, q, v, a))

    def link_transforms(self, q_deg) -> dict[str, list[float]]:
        """Every frame's pose as a row-major 4x4, for a viewer to draw.

        Forward kinematics rather than TF: the picture is then guaranteed to
        agree with the model the collision check and the regression use, which
        is the whole point of showing it.
        """
        q = np.radians(np.asarray(q_deg, dtype=float))
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        poses = {}
        for index, frame in enumerate(self.model.frames):
            poses[frame.name] = self.data.oMf[index].homogeneous.reshape(
                -1).tolist()
        return poses


def friction_row(velocity_deg_s: float) -> np.ndarray:
    """Coulomb, viscous and constant offset terms for one joint."""
    return np.array([np.sign(velocity_deg_s), velocity_deg_s, 1.0])


@dataclass
class JointRegression:
    """Least-squares fit of one joint's current against its own regressor."""

    joint: int
    columns: list[int]
    parameters: np.ndarray
    condition_number: float
    residual_rms_a: float
    samples: int
    effective_rank: int = 0
    holdout_rms_a: float | None = None
    friction: dict = field(default_factory=dict)
    components: ModelComponents = DEFAULT_COMPONENTS

    def as_dict(self) -> dict:
        return {
            "joint": self.joint,
            "columns": list(self.columns),
            "parameters": [float(value) for value in self.parameters],
            "condition_number": float(self.condition_number),
            "residual_rms_a": float(self.residual_rms_a),
            "holdout_rms_a": (
                None if self.holdout_rms_a is None else float(self.holdout_rms_a)),
            "effective_rank": int(self.effective_rank),
            "samples": int(self.samples),
            "friction": dict(self.friction),
            "components": self.components.as_dict(),
        }


def identifiable_columns(
    stacked: np.ndarray, tolerance: float = 1e-8,
) -> list[int]:
    """Columns that actually carry information in this data set.

    Standard base-parameter reduction: rank-revealing QR keeps the independent
    directions and drops columns the experiment cannot separate.
    """
    if stacked.size == 0:
        return []
    scale = np.linalg.norm(stacked, axis=0)
    active = np.flatnonzero(scale > tolerance * max(1.0, scale.max()))
    if active.size == 0:
        return []
    normalised = stacked[:, active] / scale[active]
    _q, r, permutation = _pivoted_qr(normalised)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return []
    keep = diagonal > tolerance * diagonal[0]
    return sorted(int(active[permutation[index]])
                  for index in np.flatnonzero(keep))


def _pivoted_qr(matrix: np.ndarray):
    try:
        from scipy.linalg import qr  # noqa: PLC0415

        q, r, permutation = qr(matrix, mode="economic", pivoting=True)
        return q, r, permutation
    except ImportError:
        # Greedy Gram-Schmidt pivoting keeps the dependency optional.
        remaining = list(range(matrix.shape[1]))
        order, basis = [], []
        residual = matrix.copy()
        while remaining:
            norms = np.linalg.norm(residual[:, remaining], axis=0)
            best = int(np.argmax(norms))
            column = remaining.pop(best)
            order.append(column)
            vector = residual[:, column]
            norm = np.linalg.norm(vector)
            if norm <= 1e-12:
                break
            vector = vector / norm
            basis.append(vector)
            residual = residual - np.outer(vector, vector @ residual)
        rank = len(basis)
        r = np.zeros((rank, matrix.shape[1]))
        for row, vector in enumerate(basis):
            r[row] = vector @ matrix
        return np.array(basis).T, r, np.array(order + remaining)


def truncated_solve(
    matrix: np.ndarray, target: np.ndarray, maximum_condition: float,
) -> tuple[np.ndarray, int, float]:
    """Least squares with the condition number capped.

    Rank alone does not make a parameter trustworthy: directions with tiny
    singular values fit the training noise and then explode elsewhere, so they
    are dropped rather than merely down-weighted.
    """
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.zeros(matrix.shape[1]), 0, np.inf
    keep = int(np.count_nonzero(singular >= singular[0] / maximum_condition))
    keep = max(1, keep)
    inverse = np.zeros_like(singular)
    inverse[:keep] = 1.0 / singular[:keep]
    solution = vt.T @ (inverse * (u.T @ target))
    condition = float(singular[0] / singular[keep - 1])
    return solution, keep, condition


LOAD_FRICTION_PASSES = 3


def _bounded_positions(columns: list[int], components: ModelComponents,
                       rigid_width: int) -> list[int]:
    """Which entries of the solution vector physics forces to be non-negative."""
    names = components.column_names()
    bounded = []
    for position, column in enumerate(columns):
        if column < rigid_width:
            continue
        if names[column - rigid_width] in NONNEGATIVE_COLUMNS:
            bounded.append(position)
    return bounded


def physical_solve(matrix: np.ndarray, target: np.ndarray,
                   maximum_condition: float, bounded: list[int],
                   ) -> tuple[np.ndarray, int, float]:
    """Least squares that cannot return a negative friction coefficient.

    The free solve runs first and is kept whenever it already lands inside
    physics, so well-posed joints are unaffected. When it does not, the two
    blocks are solved alternately: the unbounded block keeps the condition-number
    truncation that stops ill-conditioned directions exploding, and the bounded
    block is solved by non-negative least squares. Without this a Stribeck
    column that is nearly collinear with the Coulomb column is free to answer
    with a huge cancelling pair, which fits well and means nothing.
    """
    solution, rank, condition = truncated_solve(
        matrix, target, maximum_condition)
    if not bounded or np.all(solution[bounded] >= 0.0):
        return solution, rank, condition

    from scipy.optimize import nnls

    free = [index for index in range(matrix.shape[1]) if index not in bounded]
    working = np.zeros(matrix.shape[1])
    working[bounded] = np.maximum(solution[bounded], 0.0)
    free_rank = rank
    for _ in range(PHYSICAL_PASSES):
        if free:
            residual = target - matrix[:, bounded] @ working[bounded]
            values, free_rank, _condition = truncated_solve(
                matrix[:, free], residual, maximum_condition)
            working[free] = values
        residual = target - matrix[:, free] @ working[free] if free else target
        working[bounded], _norm = nnls(matrix[:, bounded], residual)
    return working, free_rank, condition


def _stack(rigid: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
           components: ModelComponents, loads: np.ndarray) -> np.ndarray:
    if not components.column_names():
        return rigid
    extra = np.array([extra_row(velocity, acceleration, components, load)
                      for velocity, acceleration, load
                      in zip(velocities, accelerations, loads)])
    return np.hstack([rigid, extra])


def _rigid_current(stacked: np.ndarray, columns: list[int],
                   solution: np.ndarray, rigid_width: int) -> np.ndarray:
    """What the rigid-body columns alone predict, used as the load estimate."""
    picks = [index for index, column in enumerate(columns)
             if column < rigid_width]
    if not picks:
        return np.zeros(stacked.shape[0])
    return stacked[:, [columns[index] for index in picks]] @ solution[picks]


def _solve(rigid: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
           components: ModelComponents, target: np.ndarray, tolerance: float,
           maximum_condition: float, rigid_width: int):
    """Least squares, repeated while a column depends on the fitted load.

    The load column is bilinear in the parameters, so one pass cannot build it.
    Fitting, re-estimating the load and refitting keeps every pass an ordinary
    least-squares problem.
    """
    loads = np.zeros(target.size)
    passes = LOAD_FRICTION_PASSES if components.load_friction else 1
    outcome = None
    for _ in range(passes):
        stacked = _stack(rigid, velocities, accelerations, components, loads)
        columns = identifiable_columns(stacked, tolerance)
        if not columns:
            return None
        bounded = _bounded_positions(columns, components, rigid_width)
        solution, rank, condition = physical_solve(
            stacked[:, columns], target, maximum_condition, bounded)
        outcome = (stacked, columns, solution, rank, condition)
        loads = _rigid_current(stacked, columns, solution, rigid_width)
    return outcome


def _fit_transition(rigid, velocities, accelerations, components, target,
                    tolerance, maximum_condition, rigid_width, mask=None):
    """Solve once per candidate reversal width and keep the best.

    The width sets the shape of a column rather than its amplitude, so it
    cannot be solved for linearly. Amplitudes are still fitted on everything;
    only the choice between widths is scored, and only on the rows the caller
    marks.

    Those rows matter. Scoring on all of the data lets the width absorb
    whatever the rigid-body block could not: on a joint whose block was
    truncated to rank eleven of nineteen, the search cut the residual eightfold
    by moving the width from the planted 1.8 to 2.44 and inflating Coulomb by a
    third. Scored on passes where the joint sweeps about one pose, gravity is a
    constant the offset takes and speed is the only thing that varies, which is
    the condition under which a reversal width means anything.
    """
    candidates = tuple(getattr(components, "coulomb_transition_search", ())
                       or ())
    default = replace(components, coulomb_transition_search=())
    baseline = _solve(rigid, velocities, accelerations, default, target,
                      tolerance, maximum_condition, rigid_width)
    if not candidates or baseline is None:
        return None if baseline is None else (default, baseline)
    if mask is None or int(np.count_nonzero(mask)) < MINIMUM_TRANSITION_SAMPLES:
        return (default, baseline)

    mask = np.asarray(mask, dtype=bool)
    speeds = np.abs(np.asarray(velocities, dtype=float))[mask]
    reference = _residual_of(baseline, target, mask)
    best = None
    for width in candidates:
        # Below this the tanh column is saturated at every sample present, so
        # nothing distinguishes one such candidate from another.
        if int(np.count_nonzero(speeds <= 3.0 * width)) < 4:
            continue
        trial = replace(default, coulomb_transition_deg_s=float(width))
        outcome = _solve(rigid, velocities, accelerations, trial, target,
                         tolerance, maximum_condition, rigid_width)
        if outcome is None:
            continue
        residual = _residual_of(outcome, target, mask)
        if best is None or residual < best[0]:
            best = (residual, trial, outcome)
    if best is None or best[0] > reference * (1.0 - TRANSITION_GAIN):
        return (default, baseline)
    return (best[1], best[2])


def _residual_of(outcome, target, mask=None) -> float:
    stacked, columns, solution, _rank, _condition = outcome
    error = target - stacked[:, columns] @ solution
    if mask is not None:
        error = error[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.mean(error ** 2)))


def fit_joint(
    joint: int, regressors: list[np.ndarray], velocities: list[float],
    currents: list[float], include_friction: bool = True,
    tolerance: float = 1e-6, maximum_condition: float = 1.0e3,
    holdout_fraction: float = 0.25, seed: int = 0,
    accelerations: list[float] | None = None,
    components: ModelComponents | None = None,
    transition_rows: list[bool] | None = None,
) -> JointRegression:
    """Regress one joint's measured current onto its dynamics columns."""
    if components is None:
        components = DEFAULT_COMPONENTS if include_friction else ModelComponents(
            friction=False, offset=False)
    accelerations = (
        [0.0] * len(velocities) if accelerations is None else list(accelerations))
    extra_names = components.column_names()

    rigid = np.asarray(
        [np.asarray(regressor[joint], dtype=float) for regressor in regressors],
        dtype=float)
    rigid_width = rigid.shape[1]
    velocities = np.asarray(velocities, dtype=float)
    accelerations = np.asarray(accelerations, dtype=float)
    target = np.asarray(currents, dtype=float)

    outcome = _fit_transition(rigid, velocities, accelerations, components,
                              target, tolerance, maximum_condition,
                              rigid_width, transition_rows)
    if outcome is None:
        raise ValueError(f"joint{joint + 1} data does not excite any parameter")
    components, outcome = outcome
    stacked, columns, solution, rank, condition = outcome

    holdout_rms = None
    count = rigid.shape[0]
    if 0.0 < holdout_fraction < 0.5 and count >= 20:
        rng = np.random.default_rng(seed)
        order = rng.permutation(count)
        split = int(count * (1.0 - holdout_fraction))
        train, test = order[:split], order[split:]
        trial = _solve(rigid[train], velocities[train], accelerations[train],
                       components, target[train], tolerance, maximum_condition,
                       rigid_width)
        if trial is not None:
            _unused, trial_columns, trial_solution, _rank, _cond = trial
            # The held-out load has to come from the training fit, otherwise the
            # holdout is scored with parameters that already saw it.
            blank = _stack(rigid[test], velocities[test], accelerations[test],
                           components, np.zeros(test.size))
            loads = _rigid_current(
                blank, trial_columns, trial_solution, rigid_width)
            scored = _stack(rigid[test], velocities[test], accelerations[test],
                            components, loads)
            error = scored[:, trial_columns] @ trial_solution - target[test]
            holdout_rms = float(np.sqrt(np.mean(error ** 2)))

    residual = float(
        np.sqrt(np.mean((target - stacked[:, columns] @ solution) ** 2)))

    friction = {}
    if extra_names:
        width = stacked.shape[1]
        for offset, name in enumerate(extra_names):
            column = width - len(extra_names) + offset
            if column in columns:
                friction[name] = float(solution[columns.index(column)])
    return JointRegression(
        joint=joint, columns=columns, parameters=solution,
        condition_number=condition, residual_rms_a=residual,
        samples=len(currents), effective_rank=rank,
        holdout_rms_a=holdout_rms, friction=friction, components=components,
    )


def predict_joint(regression: JointRegression, regressor: np.ndarray,
                  velocity: float, include_friction: bool = True,
                  acceleration: float = 0.0) -> float:
    rigid = np.asarray(regressor[regression.joint], dtype=float)
    components = regression.components
    if not components.column_names():
        return float(rigid[regression.columns] @ regression.parameters)
    extra = extra_row(velocity, acceleration, components)
    row = np.concatenate([rigid, extra])
    if components.load_friction:
        load = _rigid_current(
            row[None, :], regression.columns, regression.parameters,
            rigid.size)[0]
        row = np.concatenate(
            [rigid, extra_row(velocity, acceleration, components, load)])
    if not include_friction:
        # Zeroing the friction block is what lets a caller separate the
        # rigid-body current from what friction adds on top of it.
        row = np.concatenate([rigid, np.zeros_like(extra)])
    return float(row[regression.columns] @ regression.parameters)


def stacked_condition_number(rows: list[np.ndarray]) -> float:
    """Conditioning of a candidate experiment, the standard design criterion.

    Scored after base-parameter reduction: columns the experiment cannot excite
    would otherwise dominate the ratio and hide real differences in quality.
    """
    if not rows:
        return np.inf
    stacked = np.vstack(rows)
    columns = identifiable_columns(stacked)
    if not columns:
        return np.inf
    reduced = stacked[:, columns]
    scale = np.linalg.norm(reduced, axis=0)
    singular = np.linalg.svd(reduced / scale, compute_uv=False)
    return float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
