"""Physical plausibility of identified inertial parameters.

This module *checks*; it does not project. Projecting onto the physically
consistent set is a semidefinite program and needs a solver this workspace does
not have, so claiming to do it would be worse than not doing it.

The check survives the current-domain scaling. Each joint's parameters are the
true ones divided by that joint's unknown torque constant, and that constant is
positive, so the pseudo-inertia matrix is scaled by a positive number - which
leaves positive-definiteness, and therefore this verdict, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARAMETERS_PER_LINK = 10


@dataclass
class LinkVerdict:
    """Whether one link's ten parameters could describe a real rigid body."""

    link: int
    feasible: bool
    mass: float
    minimum_eigenvalue: float
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "link": self.link,
            "feasible": self.feasible,
            "mass": round(float(self.mass), 9),
            "minimum_eigenvalue": round(float(self.minimum_eigenvalue), 12),
            "reasons": list(self.reasons),
        }


def pseudo_inertia(parameters) -> np.ndarray:
    """The 4x4 pseudo-inertia J(phi) whose positive-definiteness is the test.

    ``parameters`` is one link's ten values in Pinocchio order:
    ``[m, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]``.
    """
    values = np.asarray(parameters, dtype=float)
    if values.size != PARAMETERS_PER_LINK:
        raise ValueError(f"expected {PARAMETERS_PER_LINK} values, got {values.size}")
    mass = values[0]
    h = values[1:4]
    ixx, ixy, iyy, ixz, iyz, izz = values[4:10]
    inertia = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ])
    # Second moment of mass, which is what must be positive semidefinite.
    sigma = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = sigma
    matrix[:3, 3] = h
    matrix[3, :3] = h
    matrix[3, 3] = mass
    return matrix


def check_link(parameters, link: int = 0, tolerance: float = 1e-9) -> LinkVerdict:
    values = np.asarray(parameters, dtype=float)
    reasons: list[str] = []
    if not np.all(np.isfinite(values)):
        return LinkVerdict(link, False, float("nan"), float("nan"),
                           ("parameters contain a non-finite value",))

    mass = float(values[0])
    if mass <= 0.0:
        reasons.append(f"mass {mass:.6g} is not positive")

    matrix = pseudo_inertia(values)
    eigenvalues = np.linalg.eigvalsh(matrix)
    smallest = float(eigenvalues.min())
    if smallest < -abs(tolerance):
        reasons.append(
            f"pseudo-inertia is not positive semidefinite "
            f"(smallest eigenvalue {smallest:.3e})")

    return LinkVerdict(
        link=link, feasible=not reasons, mass=mass,
        minimum_eigenvalue=smallest, reasons=tuple(reasons))


def check_parameters(parameters, tolerance: float = 1e-9) -> list[LinkVerdict]:
    """Check a full parameter vector, ten values per link."""
    values = np.asarray(parameters, dtype=float)
    if values.size % PARAMETERS_PER_LINK:
        raise ValueError(
            f"parameter vector of length {values.size} is not a whole number "
            f"of links")
    links = values.size // PARAMETERS_PER_LINK
    return [
        check_link(values[index * PARAMETERS_PER_LINK:
                          (index + 1) * PARAMETERS_PER_LINK], index, tolerance)
        for index in range(links)
    ]


def summarise(verdicts: list[LinkVerdict]) -> dict:
    infeasible = [v for v in verdicts if not v.feasible]
    return {
        "links": len(verdicts),
        "feasible": not infeasible,
        "infeasible_links": [v.link for v in infeasible],
        "detail": [v.as_dict() for v in verdicts],
        "projection": (
            "not implemented: projecting onto the physically consistent set "
            "requires a semidefinite solver (picos/cvxopt), which is absent here"
        ),
    }
