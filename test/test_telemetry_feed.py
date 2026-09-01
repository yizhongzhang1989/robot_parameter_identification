"""The live-signal feed: every frame the robot published, not every poll's worth.

Polling for the newest sample aliases a fast signal into a different one with
the peaks removed, so the bridge keeps a short history and the panel collects
it by cursor.
"""

import unittest

try:
    from robot_parameter_identification.dashboard.http_server import build_routes
    from robot_parameter_identification.dashboard.service import (
        DashboardConfig, IdentificationService)
    from fixtures import synthetic_urdf, test_profile, PREFIX
except ImportError as error:
    raise unittest.SkipTest(f"needs pinocchio: {error}") from error

JOINTS = [f"{PREFIX}joint{index}" for index in range(1, 8)]


class Bridge:
    """The part of the node the service asks for telemetry."""

    def __init__(self, frames: int = 0) -> None:
        self.frames = [{"position_deg": [float(step)] * 7} for step in range(frames)]

    def latest_sample(self):
        return self.frames[-1] if self.frames else None

    def history_since(self, cursor: int, limit: int = 400) -> dict:
        wanted = self.frames[cursor:]
        return {"cursor": len(self.frames),
                "dropped": max(0, len(wanted) - limit),
                "frames": [{"age_s": 0.001, **frame} for frame in wanted[-limit:]]}

    def health(self):
        return {"telemetry_ok": True, "sample_age_s": 0.0, "action_ok": True}

    def elsewhere(self):
        return {}

    def observed_signals(self):
        return set()


class OlderBridge:
    """A bridge that only ever offers the newest frame."""

    def latest_sample(self):
        return {"position_deg": [1.0] * 7}

    def health(self):
        return {"telemetry_ok": True, "sample_age_s": 0.0, "action_ok": True}

    def elsewhere(self):
        return {}

    def observed_signals(self):
        return set()


def service(frames: int = 0) -> IdentificationService:
    made = IdentificationService(DashboardConfig(), bridge=Bridge(frames),
                                 profile=test_profile())
    made.adopt_description(synthetic_urdf())
    made.adopt_driven_joints(JOINTS)
    return made


class TelemetryFeedTest(unittest.TestCase):
    def test_a_fresh_panel_is_handed_the_whole_window(self):
        payload = service(50).telemetry_since(0)
        self.assertEqual(len(payload["frames"]), 50)
        self.assertEqual(payload["cursor"], 50)
        self.assertEqual(payload["joint_names"], JOINTS)

    def test_a_second_poll_only_gets_what_it_has_not_seen(self):
        made = service(50)
        first = made.telemetry_since(0)
        made.bridge.frames.append({"position_deg": [99.0] * 7})
        second = made.telemetry_since(first["cursor"])
        self.assertEqual(len(second["frames"]), 1)
        self.assertEqual(second["frames"][0]["position_deg"][0], 99.0)

    def test_falling_behind_is_reported_rather_than_hidden(self):
        made = service(0)
        made.bridge.frames = [{"position_deg": [float(i)] * 7} for i in range(900)]
        payload = made.telemetry_since(0)
        self.assertEqual(len(payload["frames"]), 400)
        self.assertEqual(payload["dropped"], 500)
        # What survives is the newest, so the plot stays current.
        self.assertEqual(payload["frames"][-1]["position_deg"][0], 899.0)

    def test_a_bridge_with_no_history_still_yields_the_latest_frame(self):
        made = service(3)
        made.bridge = OlderBridge()
        payload = made.telemetry_since(0)
        self.assertEqual(len(payload["frames"]), 1)

    def test_no_bridge_at_all_is_answered_not_raised(self):
        made = IdentificationService(DashboardConfig(), profile=test_profile())
        self.assertEqual(made.telemetry_since(0)["frames"], [])


class QueryRouteTest(unittest.TestCase):
    def test_the_cursor_arrives_as_a_query_string(self):
        made = service(10)
        route = build_routes(made)["/api/telemetry"]
        self.assertEqual(route[0], "GET")
        # The HTTP layer hands GET handlers strings, exactly as they parse.
        self.assertEqual(len(route[1]({"since": "7"})["frames"]), 3)
        self.assertEqual(len(route[1]({})["frames"]), 10)


if __name__ == "__main__":
    unittest.main()
