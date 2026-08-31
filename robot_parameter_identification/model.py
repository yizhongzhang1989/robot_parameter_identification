"""Optional model columns beyond the rigid-body regressor.

Each component is a feature flag, so the model structure is a configuration
choice rather than a code change. Every column here stays *linear in its
parameter*, which is what keeps the fit an ordinary least-squares problem.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class ModelComponents:
    """Which non-rigid-body terms the per-joint model carries."""

    friction: bool = True
    offset: bool = True
    # Velocity scale over which the Coulomb term reverses. Zero means a hard
    # sign(), which asserts the joint is already at full Coulomb friction the
    # instant it moves. On a harmonic drive the reversal is spread over a
    # finite speed, and most of this arm's data sits inside that spread.
    coulomb_transition_deg_s: float = 0.0
    # Candidate widths to choose between, per joint. Twenty rungs of hardware
    # data put the best width between 0.28 and 1.37 deg/s depending on the
    # joint, so one asserted figure cannot serve them all: holding 1.8 for
    # every joint cost the worst one sixty per cent of its held-out error.
    coulomb_transition_search: tuple[float, ...] = ()
    actuator_inertia: bool = False
    stribeck: bool = False
    # Offer the column and let each joint's data decide. Measured on this arm:
    # it earns its place on the three loaded pitch joints and costs the four
    # roll joints, so switching it on or off for the whole arm is wrong either
    # way. On the joints that do not want it the non-negativity constraint
    # already drives it to zero; the selection stops the ones that land on a
    # small positive value by luck from keeping it.
    stribeck_search: bool = False
    # Stribeck is only linear once this is fixed, so it is a setting, not a fit.
    stribeck_speed_deg_s: float = 2.0
    # Candidate decay widths. Each candidate remains a linear fit; grouped
    # validation chooses the width instead of asserting one for every joint.
    stribeck_speed_search: tuple[float, ...] = ()
    # Friction that grows with transmitted load, as a gearbox's efficiency loss
    # would. Its column needs a load estimate, which is itself being fitted, so
    # this one costs an iteration.
    load_friction: bool = False
    # Offer it and let each joint decide, as with Stribeck. It was measured off
    # by default because every sweep used to be taken at one posture, where a
    # joint's load barely moves and the column had nothing to fit.
    load_friction_search: bool = False
    # The static-to-dynamic excess may itself grow with transmitted load. This
    # is a separate column from load_friction because it decays with speed.
    load_stribeck: bool = False
    load_stribeck_search: bool = False

    def column_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.friction:
            names.extend(("coulomb", "viscous"))
        if self.stribeck:
            names.append("stribeck")
        if self.load_friction:
            names.append("load_friction")
        if self.load_stribeck:
            names.append("load_stribeck")
        if self.actuator_inertia:
            names.append("actuator_inertia")
        if self.offset:
            names.append("offset")
        return tuple(names)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ModelComponents":
        data = dict(payload or {})
        known = {field: data[field] for field in cls.__dataclass_fields__
                 if field in data}
        if "stribeck_speed_deg_s" in known:
            speed = float(known["stribeck_speed_deg_s"])
            if not np.isfinite(speed) or speed <= 0.0:
                raise ValueError("stribeck_speed_deg_s must be positive")
            known["stribeck_speed_deg_s"] = speed
        if "stribeck_speed_search" in known:
            speeds = tuple(float(value) for value in
                           known["stribeck_speed_search"] or ())
            if any(not np.isfinite(value) or value <= 0.0 for value in speeds):
                raise ValueError("stribeck_speed_search values must be positive")
            known["stribeck_speed_search"] = speeds
        if "coulomb_transition_deg_s" in known:
            width = float(known["coulomb_transition_deg_s"])
            if not np.isfinite(width) or width < 0.0:
                raise ValueError(
                    "coulomb_transition_deg_s must be zero or positive")
            known["coulomb_transition_deg_s"] = width
        if "coulomb_transition_search" in known:
            widths = tuple(float(w) for w in
                           known["coulomb_transition_search"] or ())
            if any(not np.isfinite(w) or w <= 0.0 for w in widths):
                raise ValueError(
                    "coulomb_transition_search widths must be positive")
            known["coulomb_transition_search"] = widths
        return cls(**known)


DEFAULT_COMPONENTS = ModelComponents()


def extra_row(velocity_deg_s: float, acceleration_deg_s2: float = 0.0,
              components: ModelComponents = DEFAULT_COMPONENTS,
              load_a: float = 0.0) -> np.ndarray:
    """The non-rigid-body columns for one joint at one instant."""
    values: list[float] = []
    if components.friction:
        width = components.coulomb_transition_deg_s
        shape = (np.tanh(velocity_deg_s / width) if width > 0.0
                 else np.sign(velocity_deg_s))
        values.extend((float(shape), float(velocity_deg_s)))
    if components.stribeck:
        decay = np.exp(-abs(velocity_deg_s) / components.stribeck_speed_deg_s)
        values.append(float(np.sign(velocity_deg_s) * decay))
    if components.load_friction:
        values.append(float(np.sign(velocity_deg_s) * abs(load_a)))
    if components.load_stribeck:
        decay = np.exp(-abs(velocity_deg_s) / components.stribeck_speed_deg_s)
        values.append(float(np.sign(velocity_deg_s) * decay * abs(load_a)))
    if components.actuator_inertia:
        # Reflected rotor and gearbox inertia. Without this column it is
        # absorbed into the link inertia, which then stops being a link inertia.
        values.append(float(acceleration_deg_s2))
    if components.offset:
        values.append(1.0)
    return np.asarray(values, dtype=float)
