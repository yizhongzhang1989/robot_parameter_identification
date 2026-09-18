"""Bounded-memory current evidence from the full position-control telemetry stream."""

import math
import threading


class CurrentMeasurements:
    def __init__(self, channel="current_a"):
        self.channel = channel
        self._joints = []
        self._lock = threading.Lock()

    @staticmethod
    def _finite(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def observe(self, sample):
        with self._lock:
            for index, raw in enumerate(sample.get(self.channel) or []):
                while len(self._joints) <= index:
                    self._joints.append({
                        "samples": 0, "invalid_samples": 0, "minimum_a": None,
                        "maximum_a": None, "peak_abs_a": None, "mean_square": 0.0,
                        "peak_stamp_s": None,
                    })
                entry = self._joints[index]
                value = self._finite(raw)
                if value is None:
                    entry["invalid_samples"] += 1
                    continue
                entry["samples"] += 1
                entry["mean_square"] += (
                    value * value - entry["mean_square"]) / entry["samples"]
                entry["minimum_a"] = (value if entry["minimum_a"] is None
                                      else min(value, entry["minimum_a"]))
                entry["maximum_a"] = (value if entry["maximum_a"] is None
                                      else max(value, entry["maximum_a"]))
                if entry["peak_abs_a"] is None or abs(value) > entry["peak_abs_a"]:
                    entry["peak_abs_a"] = abs(value)
                    entry["peak_stamp_s"] = self._finite(sample.get("stamp_s"))

    def as_dict(self, names=()):
        with self._lock:
            joints = []
            for index, values in enumerate(self._joints):
                entry = dict(values)
                mean_square = entry.pop("mean_square")
                entry["rms_a"] = math.sqrt(mean_square) if entry["samples"] else None
                entry["joint"] = names[index] if index < len(names) else f"joint{index + 1}"
                joints.append(entry)
            return {"policy": "record_only", "source": "full_telemetry",
                    "channel": self.channel, "unit": "A", "joints": joints}
