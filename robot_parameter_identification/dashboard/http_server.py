"""Static assets, the JSON API, and a mesh proxy for the 3D view."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import json
import mimetypes
import threading

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY = 4 << 20


def build_routes(service, node=None) -> dict:
    """path -> (method, handler). Handlers take the decoded JSON body."""
    return {
        "/api/state": ("GET", lambda _body: service.snapshot()),
        "/api/activity": ("GET", lambda _body: service.activity_payload()),
        "/api/telemetry": ("GET",
                           lambda query: service.telemetry_since(
                               query.get("since", 0))),
        "/api/runs": ("GET", lambda _body: {"runs": service.runs()}),
        "/api/reports": ("GET", lambda _body: service.reports_payload()),
        "/api/viewer": ("GET", lambda _body: _viewer(service, node)),
        "/api/preview": ("GET", lambda _body: service.preview_payload()),
        "/api/workspace": ("GET", lambda _body: service.workspace_payload()),
        "/api/workspace/set": ("POST",
                               lambda body: service.set_workspace_range(
                                   body["range_deg"])
                               if "range_deg" in body
                               else service.set_workspace_limit(
                                   body.get("limit_deg"))),
        "/api/plan": ("POST",
                      lambda body: service.plan_preview(
                          str(body.get("mode", "gravity")),
                          body.get("options") or {})),
        "/api/rescreen": ("POST", lambda _body: service.rescreen()),
        "/api/kinematics": ("POST", lambda body: service.kinematics(body)),
        "/api/profile": ("GET", lambda _body: service.profile_payload()),
        "/api/profile/apply": ("POST",
                               lambda body: service.apply_profile(
                                   body.get("profile") or {})),
        "/api/profile/save": ("POST",
                              lambda body: service.save_profile(
                                  str(body.get("name", "")))),
        "/api/profile/reset": ("POST", lambda _body: service.reset_profile()),
        "/api/obstacles": ("POST", lambda body: _obstacles(service, body)),
        "/api/config/save": ("POST",
                             lambda body: service.save_config(
                                 str(body.get("name", "")))),
        "/api/collision": ("POST",
                           lambda body: service.collision_report(
                               body.get("pose_deg"))),
        "/api/campaign": ("POST",
                          lambda body: service.start(
                              str(body.get("mode", "rehearsal")),
                              body.get("options") or {})),
        "/api/gravity-test": ("POST",
                              lambda body: service.start_gravity_test(
                                  str(body.get("mode", "")),
                                  body.get("options") or {})),
        "/api/home": ("POST", lambda _body: service.home()),
        "/api/jog": ("POST", lambda body: service.jog(body)),
        "/api/pause": ("POST", lambda _body: service.pause()),
        "/api/resume": ("POST", lambda _body: service.resume()),
        "/api/stop": ("POST", lambda _body: service.stop()),
    }


def _viewer(service, node) -> dict:
    payload = service.viewer_state()
    payload["visuals"] = node.visuals() if node is not None else []
    return payload


def _obstacles(service, body: dict):
    """One endpoint, several verbs, because the UI edits a list not a resource."""
    action = str(body.get("action", "replace"))
    if action == "add":
        return {"ok": True, "obstacle": service.add_obstacle(
            body.get("obstacle") or {})}
    if action == "update":
        return {"ok": True, "obstacle": service.update_obstacle(
            str(body.get("id", "")), body.get("changes") or {})}
    if action == "remove":
        service.remove_obstacle(str(body.get("id", "")))
        return {"ok": True}
    if action == "clear":
        return {"ok": True, "obstacles": service.replace_obstacles([])}
    return {"ok": True,
            "obstacles": service.replace_obstacles(body.get("obstacles") or [])}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Every request would otherwise be echoed to the node's stderr.
    def log_message(self, fmt, *args) -> None:  # noqa: A003
        return

    def handle_one_request(self) -> None:
        # A browser closing a poll mid-flight is routine; the base class logs it
        # as an unhandled exception and buries the real messages.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/mesh":
            return self._mesh(parse_qs(parsed.query))
        if path.startswith("/runs/"):
            return self._run_file(path[len("/runs/"):])
        route = self.server.routes.get(path)
        if route and route[0] == "GET":
            # Query parameters reach a GET handler in the same shape a POST
            # body does, so handlers never learn which verb carried them.
            query = {key: values[0]
                     for key, values in parse_qs(parsed.query).items()}
            return self._json_result(route[1], query)
        return self._static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        route = self.server.routes.get(path)
        if not route or route[0] != "POST":
            return self._send(404, b"no such endpoint", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._send(413, b"body too large", "text/plain")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, UnicodeDecodeError) as error:
            return self._json(400, {"ok": False, "message": str(error)})
        return self._json_result(route[1], body)

    def _json_result(self, handler, body: dict) -> None:
        try:
            payload = handler(body)
        except (KeyError, ValueError, RuntimeError) as error:
            # Operator mistakes -- an unknown frame, an edit during a run --
            # are answered, not raised: the dashboard shows the sentence.
            return self._json(400, {"ok": False, "message": str(error)})
        except Exception as error:  # noqa: BLE001
            return self._json(500, {"ok": False, "message": repr(error)})
        if isinstance(payload, dict) and "ok" not in payload:
            payload = dict(payload)
            payload["ok"] = True
        return self._json(200, payload)

    def _json(self, status: int, payload) -> None:
        body = json.dumps(payload, default=_plain).encode("utf-8")
        self._send(status, body, "application/json")

    def _static(self, path: str) -> None:
        relative = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self._send(403, b"forbidden", "text/plain")
        if not target.is_file():
            return self._send(404, b"not found", "text/plain")
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".js":
            kind = "text/javascript"
        self._send(200, target.read_bytes(), kind)

    def _run_file(self, relative: str) -> None:
        """Serve a saved run's report and data out of the output directory.

        Resolved before it is checked, so neither ".." nor a symlink planted in
        the folder can reach a file outside it.
        """
        service = getattr(self.server, "service", None)
        if service is None:
            return self._send(404, b"no results", "text/plain")
        root = Path(service.config.output_directory).resolve()
        target = (root / unquote(relative)).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return self._send(403, b"forbidden", "text/plain")
        if not target.is_file():
            return self._send(404, b"not found", "text/plain")
        kind = mimetypes.guess_type(target.name)[0] or "text/plain"
        self._send(200, target.read_bytes(), kind)

    def _mesh(self, query: dict) -> None:
        """Serve a URDF mesh by package name, so the browser can load it."""
        resolver = getattr(self.server, "mesh_resolver", None)
        if resolver is None:
            return self._send(404, b"no mesh resolver", "text/plain")
        package = (query.get("pkg") or [""])[0]
        relative = unquote((query.get("path") or [""])[0])
        try:
            data, kind = resolver(package, relative)
        except (KeyError, ValueError, OSError) as error:
            return self._send(404, str(error).encode("utf-8"), "text/plain")
        self._send(200, data, kind)

    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return str(value)


class DashboardServer:
    """The web surface, on its own thread."""

    def __init__(self, service, port: int = 8300, host: str = "0.0.0.0",
                 mesh_resolver=None, node=None) -> None:
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.routes = build_routes(service, node)
        self.httpd.mesh_resolver = mesh_resolver
        self.httpd.service = service
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True, name="dashboard-http")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
