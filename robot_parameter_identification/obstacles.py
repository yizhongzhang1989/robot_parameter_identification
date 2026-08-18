"""Boxes bolted to robot links, and the collision query that uses them.

The identification proposes poses the arm has never been in. Something has to
veto the ones that would hit the stand, the bench or a fixture, and the robot
cannot be asked about a pose it has not reached. So the check runs on the model.

Binding is by *frame*, which is what makes this useful. A box pinned to the base
frame is a bench; a box pinned to a distal link is a tool or a payload shroud and
travels with the arm. Either way the box's pose is stored relative to its parent
frame and the kinematics place it, so the operator never has to do the algebra.

Geometry is coal (Pinocchio's collision backend), so this file adds no
dependency beyond the one the dynamics already needs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import itertools
import json
import uuid

import numpy as np
import pinocchio as pin

# Links adjacent to a box's parent are skipped: a shroud bolted to link 6 is
# always touching link 6, and often link 5, by construction. Reporting that as a
# collision would veto every pose.
NEIGHBOUR_DEPTH = 1

# Written into every saved file. Bump it only when old files stop loading.
SCHEMA_VERSION = 1
BOX_SHAPE = "box"
# Extension point: adding a shape means adding it here and to _geometry_for.
SHAPES = (BOX_SHAPE,)
MINIMUM_SIZE_M = 1e-4


def _se3(xyz_m, rpy_deg) -> pin.SE3:
    roll, pitch, yaw = np.radians(np.asarray(rpy_deg, dtype=float))
    rotation = (pin.utils.rotate("z", yaw)
                @ pin.utils.rotate("y", pitch)
                @ pin.utils.rotate("x", roll))
    return pin.SE3(rotation, np.asarray(xyz_m, dtype=float))


@dataclass
class Obstacle:
    """One axis-aligned box, posed in the frame it is bolted to."""

    parent_frame: str
    size_m: tuple[float, float, float] = (0.1, 0.1, 0.1)
    xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ""
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # Every stored obstacle names its shape so a file written today still
    # loads once cylinders or meshes exist; an unknown shape is refused rather
    # than silently read as a box.
    shape: str = BOX_SHAPE

    def __post_init__(self) -> None:
        self.size_m = tuple(float(value) for value in self.size_m)
        self.xyz_m = tuple(float(value) for value in self.xyz_m)
        self.rpy_deg = tuple(float(value) for value in self.rpy_deg)
        if len(self.size_m) != 3 or len(self.xyz_m) != 3 or len(self.rpy_deg) != 3:
            raise ValueError("size, position and orientation are 3-vectors")
        if min(self.size_m) < MINIMUM_SIZE_M:
            raise ValueError(
                f"every side must be at least {MINIMUM_SIZE_M} m, "
                f"got {self.size_m}")
        if not str(self.parent_frame).strip():
            raise ValueError("an obstacle must name the frame it is bolted to")
        if self.shape not in SHAPES:
            raise ValueError(
                f"unknown obstacle shape {self.shape!r}; this build understands "
                f"{SHAPES}")
        self.name = self.name or f"box_{self.id}"

    def placement(self) -> pin.SE3:
        return _se3(self.xyz_m, self.rpy_deg)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Obstacle":
        data = dict(payload or {})
        known = {key: data[key] for key in cls.__dataclass_fields__
                 if key in data}
        return cls(**known)


class ObstacleScene:
    """The obstacle set, and the collision query built from it.

    Rebuilding the whole geometry model on every edit is deliberate: edits come
    from a human dragging a box, so they are rare, and a stale collision model
    is the kind of bug that only shows up as a dent in the hardware.
    """

    def __init__(self, model: pin.Model, urdf_text: str = "",
                 package_dirs: list[str] | None = None) -> None:
        self.model = model
        self.urdf_text = urdf_text
        self.package_dirs = [str(path) for path in (package_dirs or [])]
        self._obstacles: dict[str, Obstacle] = {}
        self._robot_geometry: pin.GeometryModel | None = None
        self._robot_geometry_error = ""
        self._geometry: pin.GeometryModel | None = None
        self._geometry_data = None
        self._data = model.createData()
        self._obstacle_geom_ids: dict[str, int] = {}
        self._load_robot_geometry()

    # -- frames ----------------------------------------------------------

    def frame_names(self) -> list[str]:
        """Frames an obstacle may be bolted to, in kinematic order."""
        return [frame.name for frame in self.model.frames
                if frame.type in (pin.FrameType.BODY, pin.FrameType.FIXED_JOINT,
                                  pin.FrameType.JOINT)]

    def _frame_id(self, name: str) -> int:
        if not self.model.existFrame(name):
            raise KeyError(f"no frame named {name!r} in this robot")
        return self.model.getFrameId(name)

    # -- editing ---------------------------------------------------------

    def add(self, obstacle: Obstacle) -> Obstacle:
        self._frame_id(obstacle.parent_frame)
        self._obstacles[obstacle.id] = obstacle
        self._invalidate()
        return obstacle

    def update(self, obstacle_id: str, **changes) -> Obstacle:
        current = self._obstacles.get(obstacle_id)
        if current is None:
            raise KeyError(f"no obstacle {obstacle_id!r}")
        payload = current.as_dict()
        payload.update({key: value for key, value in changes.items()
                        if key in Obstacle.__dataclass_fields__})
        payload["id"] = obstacle_id
        replacement = Obstacle.from_dict(payload)
        self._frame_id(replacement.parent_frame)
        self._obstacles[obstacle_id] = replacement
        self._invalidate()
        return replacement

    def remove(self, obstacle_id: str) -> None:
        if self._obstacles.pop(obstacle_id, None) is None:
            raise KeyError(f"no obstacle {obstacle_id!r}")
        self._invalidate()

    def clear(self) -> None:
        self._obstacles.clear()
        self._invalidate()

    def replace_all(self, payloads: list[dict]) -> None:
        rebuilt = {}
        for payload in payloads or []:
            obstacle = Obstacle.from_dict(payload)
            self._frame_id(obstacle.parent_frame)
            rebuilt[obstacle.id] = obstacle
        self._obstacles = rebuilt
        self._invalidate()

    def obstacles(self) -> list[Obstacle]:
        return list(self._obstacles.values())

    def as_list(self) -> list[dict]:
        return [obstacle.as_dict() for obstacle in self._obstacles.values()]

    # -- persistence -----------------------------------------------------

    def as_document(self) -> dict:
        """The scene in the form written to disk.

        Versioned and shape-tagged so a file saved by this build survives new
        obstacle kinds, and so a newer file is refused rather than misread.
        """
        return {"schema_version": SCHEMA_VERSION,
                "obstacles": self.as_list()}

    def load_document(self, document: dict) -> list[str]:
        """Replace the scene from a saved document, reporting what was dropped.

        Obstacles bolted to frames this robot does not have are skipped rather
        than fatal: the same file may be shared between arms, and losing one
        box should not cost the operator the rest of the scene.
        """
        payload = dict(document or {})
        version = int(payload.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"obstacle file is schema version {version}; this build "
                f"understands up to {SCHEMA_VERSION}")
        kept, skipped = {}, []
        for entry in payload.get("obstacles") or []:
            try:
                obstacle = Obstacle.from_dict(entry)
                self._frame_id(obstacle.parent_frame)
            except (KeyError, ValueError) as error:
                skipped.append(str(error))
                continue
            kept[obstacle.id] = obstacle
        self._obstacles = kept
        self._invalidate()
        return skipped

    def save(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename, so an interrupted save cannot
        # leave the operator with a truncated scene.
        scratch = target.with_suffix(target.suffix + ".partial")
        scratch.write_text(json.dumps(self.as_document(), indent=2) + "\n")
        scratch.replace(target)

    def load(self, path) -> list[str]:
        return self.load_document(json.loads(Path(path).read_text()))

    def _invalidate(self) -> None:
        self._geometry = None
        self._geometry_data = None
        self._obstacle_geom_ids = {}

    # -- geometry --------------------------------------------------------

    def _load_robot_geometry(self) -> None:
        """The arm's own collision shapes, if the URDF ships any."""
        if not self.urdf_text:
            self._robot_geometry_error = "no URDF supplied"
            return
        try:
            self._robot_geometry = pin.buildGeomFromUrdfString(
                self.model, self.urdf_text, pin.GeometryType.COLLISION,
                package_dirs=self.package_dirs or None)
        except Exception as error:  # noqa: BLE001 - report, never crash the campaign
            self._robot_geometry = None
            self._robot_geometry_error = str(error)

    def geometry_report(self) -> dict:
        """What the collision check can and cannot see, for the operator."""
        robot_shapes = (0 if self._robot_geometry is None
                        else self._robot_geometry.ngeoms)
        return {
            "robot_shapes": robot_shapes,
            "obstacles": len(self._obstacles),
            "enabled_obstacles": sum(1 for item in self._obstacles.values()
                                     if item.enabled),
            "robot_geometry_error": self._robot_geometry_error,
            # Without link shapes we can still stop the arm driving a box
            # through itself, but not self-collision. Say so plainly.
            "self_collision_checked": robot_shapes > 0,
        }

    def _build(self) -> None:
        geometry = pin.GeometryModel()
        link_ids: list[int] = []
        if self._robot_geometry is not None:
            for shape in self._robot_geometry.geometryObjects:
                link_ids.append(geometry.addGeometryObject(shape))

        self._obstacle_geom_ids = {}
        for obstacle in self._obstacles.values():
            if not obstacle.enabled:
                continue
            frame_id = self._frame_id(obstacle.parent_frame)
            frame = self.model.frames[frame_id]
            # A GeometryObject hangs off a joint, so fold the frame's own offset
            # into the placement. This is what binds the box to the link.
            placement = frame.placement * obstacle.placement()
            box = _box(obstacle.size_m)
            shape = pin.GeometryObject(
                f"obstacle::{obstacle.id}", frame.parentJoint, placement, box)
            index = geometry.addGeometryObject(shape)
            self._obstacle_geom_ids[obstacle.id] = index

        self._pair(geometry, link_ids)
        self._geometry = geometry
        self._geometry_data = geometry.createData()

    def _pair(self, geometry: pin.GeometryModel, link_ids: list[int]) -> None:
        """Obstacle-vs-link and link-vs-link pairs, minus the trivial ones."""
        for obstacle in self._obstacles.values():
            index = self._obstacle_geom_ids.get(obstacle.id)
            if index is None:
                continue
            parent_joint = geometry.geometryObjects[index].parentJoint
            for link_index in link_ids:
                other = geometry.geometryObjects[link_index]
                if self._adjacent(parent_joint, other.parentJoint):
                    continue
                geometry.addCollisionPair(pin.CollisionPair(index, link_index))
            # Boxes on different links can also meet each other.
            for other_id, other_index in self._obstacle_geom_ids.items():
                if other_index <= index:
                    continue
                other_joint = geometry.geometryObjects[other_index].parentJoint
                if self._adjacent(parent_joint, other_joint):
                    continue
                geometry.addCollisionPair(pin.CollisionPair(index, other_index))

        for first, second in itertools.combinations(link_ids, 2):
            one = geometry.geometryObjects[first]
            two = geometry.geometryObjects[second]
            if self._adjacent(one.parentJoint, two.parentJoint):
                continue
            geometry.addCollisionPair(pin.CollisionPair(first, second))

    def _adjacent(self, first_joint: int, second_joint: int) -> bool:
        """True when two joints are the same or within NEIGHBOUR_DEPTH links."""
        if first_joint == second_joint:
            return True
        for start, target in ((first_joint, second_joint),
                              (second_joint, first_joint)):
            walker = int(start)
            for _ in range(NEIGHBOUR_DEPTH):
                walker = int(self.model.parents[walker])
                if walker == target:
                    return True
                if walker == 0:
                    break
        return False

    # -- queries ---------------------------------------------------------

    def _configuration(self, pose_deg) -> np.ndarray:
        angles = np.radians(np.asarray(pose_deg, dtype=float))
        if angles.size != self.model.nq:
            raise ValueError(
                f"expected {self.model.nq} joint angles, got {angles.size}")
        return angles

    def collision_free(self, pose_deg) -> bool:
        """False when this pose puts any checked pair in contact."""
        if self._geometry is None:
            self._build()
        if self._geometry.ngeoms == 0 or not self._geometry.collisionPairs:
            return True
        hit = pin.computeCollisions(
            self.model, self._data, self._geometry, self._geometry_data,
            self._configuration(pose_deg), True)
        return not hit

    def contacts(self, pose_deg) -> list[dict]:
        """Every colliding pair, named. Used to explain a rejected pose."""
        if self._geometry is None:
            self._build()
        if self._geometry.ngeoms == 0 or not self._geometry.collisionPairs:
            return []
        pin.computeCollisions(
            self.model, self._data, self._geometry, self._geometry_data,
            self._configuration(pose_deg), False)
        found = []
        for index, pair in enumerate(self._geometry.collisionPairs):
            result = self._geometry_data.collisionResults[index]
            if not result.isCollision():
                continue
            found.append({
                "first": self._geometry.geometryObjects[pair.first].name,
                "second": self._geometry.geometryObjects[pair.second].name,
            })
        return found

    def placements(self, pose_deg) -> list[dict]:
        """World pose of every enabled box, for the 3D view to draw."""
        if self._geometry is None:
            self._build()
        angles = self._configuration(pose_deg)
        pin.forwardKinematics(self.model, self._data, angles)
        pin.updateFramePlacements(self.model, self._data)
        drawn = []
        for obstacle in self._obstacles.values():
            if not obstacle.enabled:
                continue
            frame_id = self._frame_id(obstacle.parent_frame)
            world = self._data.oMf[frame_id] * obstacle.placement()
            drawn.append({
                "id": obstacle.id,
                "name": obstacle.name,
                "parent_frame": obstacle.parent_frame,
                "size_m": list(obstacle.size_m),
                "matrix": world.homogeneous.reshape(-1).tolist(),
            })
        return drawn


def _box(size_m) -> object:
    """A coal box, tolerating both the vector and the three-scalar ctor."""
    import coal  # noqa: PLC0415 - optional at import time, required here

    width, depth, height = (float(value) for value in size_m)
    try:
        return coal.Box(width, depth, height)
    except Exception:  # noqa: BLE001 - older bindings take a vector
        return coal.Box(np.array([width, depth, height], dtype=float))
