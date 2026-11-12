"""
AquaFFIA dashboard gateway (port 8001, always on, no model).

- Serves the dashboard page (web_dashboard/templates/index.html), /static and the
  continuous simulation media, mounted read-only from Deployment 1.
- Starts/stops the deployment containers so that exactly ONE model runs at a time.
  Deployment 1 is started by default when the gateway starts (i.e. after boot).

API:
    GET  /gateway/state          active deployment and status of every deployment
    POST /gateway/switch/{id}    stop the running deployment, start {id}, wait until ready

Only the containers listed in DEPLOYMENTS can be started or stopped.
Docker is controlled through /var/run/docker.sock with the standard library only.
"""

import http.client
import json
import logging
import mimetypes
import os
import socket
import threading
import time
import urllib.request

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AquaFFIAGateway")

WEB_DIR = os.environ.get("WEB_DIR", "/web")
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "/samples")
DOCKER_SOCK = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")
DEFAULT_DEPLOYMENT = int(os.environ.get("DEFAULT_DEPLOYMENT", "1"))
READY_TIMEOUT_S = float(os.environ.get("READY_TIMEOUT_S", "120"))
STOP_TIMEOUT_S = int(os.environ.get("STOP_TIMEOUT_S", "10"))

DEPLOYMENTS = {
    1: {"name": "Deployment 1", "container": "aquaffia-edge", "port": 8011},
    2: {"name": "Deployment 2", "container": "aquaffia-edge-2", "port": 8012},
    3: {"name": "Deployment 3", "container": "aquaffia-edge-3", "port": 8013},
}
if os.environ.get("DEPLOYMENTS_JSON"):
    DEPLOYMENTS = {int(k): v for k, v in json.loads(os.environ["DEPLOYMENTS_JSON"]).items()}


# ==============================================================================
# DOCKER ENGINE API (unix socket, stdlib only)
# ==============================================================================
class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def _docker(method: str, path: str, timeout: float = 30.0):
    conn = _UnixHTTPConnection(DOCKER_SOCK, timeout)
    try:
        conn.request(method, path)
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, (json.loads(body) if body and body[:1] in b"{[" else None)
    finally:
        conn.close()


def container_status(name: str) -> str:
    """running | exited | created | restarting | paused | missing"""
    status, data = _docker("GET", f"/containers/{name}/json")
    if status == 404:
        return "missing"
    if status != 200:
        return "unknown"
    return data["State"]["Status"]


def start_container(name: str) -> None:
    status, data = _docker("POST", f"/containers/{name}/start")
    if status not in (204, 304):
        raise RuntimeError(f"docker start {name}: HTTP {status} {data}")


def stop_container(name: str) -> None:
    status, data = _docker("POST", f"/containers/{name}/stop?t={STOP_TIMEOUT_S}", timeout=STOP_TIMEOUT_S + 30)
    if status not in (204, 304, 404):
        raise RuntimeError(f"docker stop {name}: HTTP {status} {data}")


def is_ready(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=timeout) as r:
            return r.status == 200 and json.loads(r.read()).get("status") == "online"
    except Exception:
        return False


# ==============================================================================
# SWITCHING LOGIC
# ==============================================================================
_switch_lock = threading.Lock()
_state = {"active": DEFAULT_DEPLOYMENT, "switching": False, "target": None, "message": ""}


def _wait_ready(dep: dict, container: str) -> bool:
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if is_ready(dep["port"]):
            return True
        if container_status(container) not in ("running", "restarting"):
            return False
        time.sleep(0.5)
    return False


def switch_to(target: int) -> dict:
    """Stop every other deployment, start `target`, wait until it is ready.
    If `target` fails to start, fall back to the default deployment."""
    if target not in DEPLOYMENTS:
        raise HTTPException(status_code=404, detail=f"Unknown deployment {target}")
    if container_status(DEPLOYMENTS[target]["container"]) == "missing":
        raise HTTPException(status_code=409, detail=f"{DEPLOYMENTS[target]['name']} is not deployed yet")
    if not _switch_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another switch is in progress")

    started = time.time()
    try:
        _state.update(switching=True, target=target, message=f"Starting {DEPLOYMENTS[target]['name']}")
        logger.info("Switching to %s", DEPLOYMENTS[target]["name"])

        # 1. Stop everything else first so that only one model is ever in memory.
        for dep_id, dep in DEPLOYMENTS.items():
            if dep_id != target and container_status(dep["container"]) in ("running", "restarting", "paused"):
                logger.info("Stopping %s (%s)", dep["name"], dep["container"])
                stop_container(dep["container"])

        # 2. Start the target and wait until /api/status answers (model loaded + warmed up).
        dep = DEPLOYMENTS[target]
        if container_status(dep["container"]) != "running":
            start_container(dep["container"])
        if _wait_ready(dep, dep["container"]):
            _state.update(active=target, message="")
            logger.info("%s ready in %.1f s", dep["name"], time.time() - started)
            return {"ok": True, "active": target, "seconds": round(time.time() - started, 1)}

        # 3. Failed: stop it and fall back to the default deployment.
        logger.error("%s failed to become ready, falling back", dep["name"])
        stop_container(dep["container"])
        fallback = DEFAULT_DEPLOYMENT
        if target != fallback:
            fb = DEPLOYMENTS[fallback]
            start_container(fb["container"])
            _wait_ready(fb, fb["container"])
        _state.update(active=fallback, message=f"{dep['name']} failed to start")
        return {"ok": False, "active": fallback, "error": f"{dep['name']} failed to start",
                "seconds": round(time.time() - started, 1)}
    finally:
        _state.update(switching=False, target=None)
        _switch_lock.release()


def current_state() -> dict:
    deployments = []
    for dep_id, dep in DEPLOYMENTS.items():
        status = container_status(dep["container"])
        deployments.append({
            "id": dep_id,
            "name": dep["name"],
            "container": dep["container"],
            "port": dep["port"],
            "status": status,
            "ready": status == "running" and is_ready(dep["port"], timeout=0.5),
        })
    return {**_state, "deployments": deployments}


# ==============================================================================
# FASTAPI APP
# ==============================================================================
app = FastAPI(title="AquaFFIA Dashboard Gateway", version="1.0.0")
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")


@app.on_event("startup")
def on_startup():
    """After boot: start the default deployment (and stop the others) in the background."""
    def _boot():
        for _ in range(30):
            try:
                switch_to(DEFAULT_DEPLOYMENT)
                return
            except HTTPException as e:
                logger.warning("Boot switch: %s", e.detail)
            except Exception as e:  # docker not ready yet
                logger.warning("Boot switch retry: %s", e)
            time.sleep(2)
    threading.Thread(target=_boot, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open(os.path.join(WEB_DIR, "templates", "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-cache"})


@app.get("/gateway/state")
def get_state():
    return current_state()


@app.post("/gateway/switch/{deployment_id}")
def post_switch(deployment_id: int):
    return switch_to(deployment_id)


@app.get("/samples/{file_path:path}")
@app.head("/samples/{file_path:path}")
def serve_sample_media(file_path: str, request: Request):
    """Same byte-range media serving as Deployment 1 (needed for video seeking)."""
    full_path = os.path.realpath(os.path.join(SAMPLES_DIR, file_path))
    real_samples = os.path.realpath(SAMPLES_DIR)
    if not full_path.startswith(real_samples) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="Sample media file not found")

    file_size = os.path.getsize(full_path)
    content_type = mimetypes.guess_type(full_path)[0] or "application/octet-stream"
    base_headers = {"Accept-Ranges": "bytes", "Content-Type": content_type}

    if request.method == "HEAD":
        return Response(status_code=200, headers={**base_headers, "Content-Length": str(file_size)})

    range_header = request.headers.get("range")
    start, end = 0, file_size - 1
    if range_header and range_header.startswith("bytes="):
        first, _, last = range_header.split("=", 1)[1].partition("-")
        start = int(first) if first else 0
        end = min(int(last), file_size - 1) if last else file_size - 1
        if start > end or start >= file_size:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    length = end - start + 1

    def body():
        with open(full_path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(512 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {**base_headers, "Content-Length": str(length)}
    if range_header:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        return StreamingResponse(body(), status_code=206, headers=headers)
    return StreamingResponse(body(), status_code=200, headers=headers)
