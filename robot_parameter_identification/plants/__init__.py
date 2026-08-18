"""Plants: the things a campaign can measure.

Imports are lazy because a rehearsal must not require ROS, and a hardware run
must not require a physics engine.
"""


def __getattr__(name):
    if name in ("AnalyticPlant", "independent_engine_available"):
        from . import analytic
        return getattr(analytic, name)
    if name in ("HardwarePlant", "HardwareConfig", "TelemetryUnavailable",
                "differentiate"):
        from . import ros_control
        return getattr(ros_control, name)
    raise AttributeError(name)


__all__ = [
    "AnalyticPlant", "independent_engine_available",
    "HardwarePlant", "HardwareConfig", "TelemetryUnavailable", "differentiate",
]
