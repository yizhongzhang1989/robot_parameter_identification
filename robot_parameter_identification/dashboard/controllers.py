"""Read-only asynchronous controller-manager inventory."""

import threading
import time


class ControllerInventory:
    """Cache manager replies without blocking telemetry or HTTP workers."""

    def __init__(self, client, request_factory, manager, clock=time.monotonic):
        self.client = client
        self.request_factory = request_factory
        self.manager = manager
        self.clock = clock
        self._lock = threading.Lock()
        self._pending = None
        self._requested_at = 0.0
        self._received_at = None
        self._received_request_at = None
        self._items = []
        self._error = "waiting"

    def poll(self):
        with self._lock:
            now = self.clock()
            if self._pending is not None:
                if self._pending.done():
                    try:
                        response = self._pending.result()
                        self._items = sorted([
                            {"name": str(item.name), "type": str(item.type),
                             "state": str(item.state),
                             "claimed_interfaces": list(item.claimed_interfaces),
                             "required_command_interfaces": list(
                                 getattr(item, "required_command_interfaces", []))}
                            for item in response.controller
                        ], key=lambda item: item["name"])
                        self._received_at = now
                        self._received_request_at = self._requested_at
                        self._error = ""
                    except Exception:  # noqa: BLE001
                        self._error = "query_failed"
                    self._pending = None
                elif now - self._requested_at >= 2.0:
                    self.client.remove_pending_request(self._pending)
                    self._pending.cancel()
                    self._pending = None
                    self._error = "timeout"
                else:
                    return
            if not self.client.service_is_ready():
                self._error = "unavailable"
                return
            try:
                self._pending = self.client.call_async(self.request_factory())
                self._requested_at = now
            except Exception:  # noqa: BLE001
                self._error = "query_failed"

    def snapshot(self):
        with self._lock:
            age = None if self._received_at is None else self.clock() - self._received_at
            available = not self._error and age is not None and age < 3.0
            return {
                "manager": self.manager,
                "available": bool(available),
                "age_s": None if age is None else round(age, 3),
                "error": self._error or ("" if available else "stale"),
                "requested_at_monotonic": self._received_request_at,
                "received_at_monotonic": self._received_at,
                "items": [dict(item, claimed_interfaces=list(item["claimed_interfaces"]),
                               required_command_interfaces=list(
                                   item["required_command_interfaces"]))
                          for item in self._items] if available else [],
            }

    def fresh_snapshot(self, timeout_s=5.0):
        started = self.clock()
        waiter = threading.Event()
        while self.clock() - started < timeout_s:
            self.poll()
            snapshot = self.snapshot()
            requested_at = snapshot["requested_at_monotonic"]
            if (snapshot["available"] and requested_at is not None and
                    requested_at >= started):
                return snapshot
            if snapshot["error"] in ("query_failed", "timeout", "unavailable"):
                raise RuntimeError(f"controller inventory query failed: {snapshot['error']}")
            waiter.wait(0.01)
        raise TimeoutError("controller inventory did not provide a post-request response")
