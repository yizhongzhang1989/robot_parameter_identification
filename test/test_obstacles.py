"""Obstacles: binding, collision, and the honesty of the geometry report."""

from pathlib import Path
import unittest

import numpy as np

try:
    import pinocchio as pin
    from robot_parameter_identification import identification as ident
    from robot_parameter_identification.obstacles import (
        Obstacle, ObstacleScene, SCHEMA_VERSION)
    from fixtures import synthetic_urdf, PREFIX
except ImportError as error:
    raise unittest.SkipTest(f"needs pinocchio: {error}") from error


def arm():
    urdf = synthetic_urdf()
    return ident.ArmModel.from_urdf_text(urdf, PREFIX), urdf


class ObstacleShapeTest(unittest.TestCase):
    def test_a_box_needs_a_parent_frame(self):
        with self.assertRaises(ValueError):
            Obstacle(parent_frame="   ")

    def test_a_flat_box_is_rejected(self):
        with self.assertRaises(ValueError):
            Obstacle(parent_frame="base_link", size_m=(0.1, 0.1, 0.0))

    def test_ids_are_unique_without_being_asked(self):
        first = Obstacle(parent_frame="base_link")
        second = Obstacle(parent_frame="base_link")
        self.assertNotEqual(first.id, second.id)

    def test_round_trip_through_a_dict(self):
        original = Obstacle(parent_frame="base_link", size_m=(0.2, 0.3, 0.4),
                            xyz_m=(1.0, 0.0, 0.5), rpy_deg=(0.0, 90.0, 0.0),
                            name="bench")
        self.assertEqual(Obstacle.from_dict(original.as_dict()), original)


class ObstacleSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm, cls.urdf = arm()
        cls.zero = np.zeros(cls.arm.joint_count)

    def scene(self):
        return ObstacleScene(self.arm.model, urdf_text=self.urdf)

    def base(self, scene):
        return f"{PREFIX}base_link"

    def test_the_urdf_supplies_link_shapes(self):
        report = self.scene().geometry_report()
        self.assertGreater(report["robot_shapes"], 0)
        self.assertTrue(report["self_collision_checked"])

    def test_a_scene_without_geometry_says_so_rather_than_pretending(self):
        blind = ObstacleScene(self.arm.model, urdf_text="")
        report = blind.geometry_report()
        self.assertEqual(report["robot_shapes"], 0)
        self.assertFalse(report["self_collision_checked"])
        self.assertTrue(report["robot_geometry_error"])

    def test_an_empty_scene_vetoes_nothing(self):
        self.assertTrue(self.scene().collision_free(self.zero))

    def test_a_distant_box_vetoes_nothing(self):
        scene = self.scene()
        scene.add(Obstacle(parent_frame=self.base(scene),
                           size_m=(0.05, 0.05, 0.05), xyz_m=(3.0, 0.0, 0.0)))
        self.assertTrue(scene.collision_free(self.zero))

    def test_a_box_around_the_arm_is_caught(self):
        scene = self.scene()
        scene.add(Obstacle(parent_frame=self.base(scene),
                           size_m=(1.5, 1.5, 1.5)))
        self.assertFalse(scene.collision_free(self.zero))

    def test_a_caught_pose_can_say_what_it_hit(self):
        scene = self.scene()
        scene.add(Obstacle(parent_frame=self.base(scene),
                           size_m=(1.5, 1.5, 1.5), name="cage"))
        contacts = scene.contacts(self.zero)
        self.assertTrue(contacts)
        self.assertTrue(any("obstacle::" in hit["first"] for hit in contacts))

    def test_disabling_a_box_takes_it_out_of_the_check(self):
        scene = self.scene()
        box = scene.add(Obstacle(parent_frame=self.base(scene),
                                 size_m=(1.5, 1.5, 1.5)))
        self.assertFalse(scene.collision_free(self.zero))
        scene.update(box.id, enabled=False)
        self.assertTrue(scene.collision_free(self.zero))

    def test_removing_a_box_takes_it_out_of_the_check(self):
        scene = self.scene()
        box = scene.add(Obstacle(parent_frame=self.base(scene),
                                 size_m=(1.5, 1.5, 1.5)))
        scene.remove(box.id)
        self.assertTrue(scene.collision_free(self.zero))

    def test_an_unknown_frame_is_refused_at_edit_time(self):
        scene = self.scene()
        with self.assertRaises(KeyError):
            scene.add(Obstacle(parent_frame="no_such_link"))

    def test_editing_an_unknown_box_is_refused(self):
        with self.assertRaises(KeyError):
            self.scene().update("nope", enabled=False)

    def test_a_box_on_a_moving_link_travels_with_it(self):
        """The whole point of frame binding: the box is not world-fixed."""
        scene = self.scene()
        moving = [name for name in scene.frame_names()
                  if name.startswith(PREFIX)][-1]
        scene.add(Obstacle(parent_frame=moving, size_m=(0.1, 0.1, 0.1),
                           xyz_m=(0.3, 0.0, 0.0), name="tool"))
        here = scene.placements(self.zero)[0]["matrix"]
        swung = self.zero.copy()
        swung[0] = 45.0
        there = scene.placements(swung)[0]["matrix"]
        self.assertFalse(np.allclose(here, there))

    def test_a_box_on_the_base_does_not_move_with_the_arm(self):
        scene = self.scene()
        scene.add(Obstacle(parent_frame=self.base(scene),
                           size_m=(0.1, 0.1, 0.1), xyz_m=(1.0, 0.0, 0.0)))
        swung = self.zero.copy()
        swung[0] = 45.0
        self.assertTrue(np.allclose(scene.placements(self.zero)[0]["matrix"],
                                    scene.placements(swung)[0]["matrix"]))

    def test_replace_all_swaps_the_scene_atomically(self):
        scene = self.scene()
        scene.add(Obstacle(parent_frame=self.base(scene)))
        scene.replace_all([Obstacle(parent_frame=self.base(scene),
                                    name="only").as_dict()])
        self.assertEqual([box.name for box in scene.obstacles()], ["only"])

    def test_a_bad_payload_leaves_the_old_scene_standing(self):
        scene = self.scene()
        keeper = scene.add(Obstacle(parent_frame=self.base(scene), name="keep"))
        with self.assertRaises(KeyError):
            scene.replace_all([{"parent_frame": "no_such_link"}])
        self.assertEqual([box.id for box in scene.obstacles()], [keeper.id])

    def test_the_wrong_number_of_angles_is_refused(self):
        with self.assertRaises(ValueError):
            self.scene().collision_free(np.zeros(3))


class PersistenceTest(ObstacleSceneTest):
    """A scene the operator drew must survive a restart."""

    def temp_path(self):
        import tempfile

        handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        handle.close()
        path = Path(handle.name)
        self.addCleanup(lambda: path.exists() and path.unlink())
        return path

    def test_a_saved_scene_reloads_identically(self):
        scene = self.scene()
        scene.add(Obstacle(parent_frame=self.base(scene), name="bench",
                           size_m=(0.4, 0.3, 0.2), xyz_m=(0.2, 0.0, -0.1)))
        path = self.temp_path()
        scene.save(path)

        restored = self.scene()
        restored.load(path)
        self.assertEqual(scene.as_list(), restored.as_list())

    def test_the_document_is_versioned_and_names_the_shape(self):
        scene = self.scene()
        scene.add(Obstacle(parent_frame=self.base(scene)))
        document = scene.as_document()
        self.assertEqual(document["schema_version"], SCHEMA_VERSION)
        self.assertEqual(document["obstacles"][0]["shape"], "box")

    def test_a_future_schema_is_refused_rather_than_misread(self):
        scene = self.scene()
        with self.assertRaises(ValueError):
            scene.load_document({"schema_version": SCHEMA_VERSION + 1,
                                 "obstacles": []})

    def test_an_unknown_shape_is_refused_rather_than_read_as_a_box(self):
        with self.assertRaises(ValueError):
            Obstacle(parent_frame="base", shape="dodecahedron")

    def test_a_box_on_a_frame_this_robot_lacks_is_skipped_not_fatal(self):
        # The same file may be shared between arms; losing one box must not
        # cost the operator the rest of the scene.
        scene = self.scene()
        good = Obstacle(parent_frame=self.base(scene), name="keep").as_dict()
        stray = dict(good, parent_frame="no_such_link", name="stray",
                     id="deadbeef")
        skipped = scene.load_document(
            {"schema_version": SCHEMA_VERSION, "obstacles": [good, stray]})
        self.assertEqual([box.name for box in scene.obstacles()], ["keep"])
        self.assertEqual(len(skipped), 1)

    def test_an_interrupted_save_leaves_no_partial_file(self):
        scene = self.scene()
        scene.add(Obstacle(parent_frame=self.base(scene)))
        path = self.temp_path()
        scene.save(path)
        self.assertFalse(path.with_suffix(path.suffix + ".partial").exists())


if __name__ == "__main__":
    unittest.main()
