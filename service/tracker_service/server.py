"""The service's one door: a websocket Home Assistant's Protect Tracker
integration connects to.

  it sends    {"type": "watch",  "camera": id, "source": <stream address>, "channel": n, "stamp": ms}
              {"type": "report", "camera": id, "packet": <one of Protect's live reports>}
              {"type": "sleep",  "camera": id}
  it is sent  {"type": "tracks", "camera": id, "mode": "auto" | "local", "packet": <boxes for one frame, in the reports' shape>}
              ("auto": what Protect reported, refined; "local": what the service sees, on its own word)
              {"type": "status", ...}   now and then

The stream address is kept in memory for as long as the camera is watched and
is never logged. There is no login here: the service listens on this machine
only (see the README), and can do nothing but look at streams it is handed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiohttp import WSMsgType, web

from .camera import Camera

log = logging.getLogger(__name__)

STATUS_EVERY = 5.0
DEFAULT_STAMP = -125.0


class Service:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.cameras: dict[str, Camera] = {}
        self._out: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/link", self.link)
        app.router.add_get("/bench", self.bench)
        app.router.add_post("/bench", self.bench)
        app.on_shutdown.append(self._shutdown)
        return app

    async def bench(self, request: web.Request) -> web.Response:
        """The model comparison (bench.py). POST starts one on the frames of
        whatever is being watched (?frames=240&every=15); GET says how far it
        is, and the numbers when it is done."""
        bench = getattr(self.engine, "bench", None)
        if bench is None:
            return web.json_response({"error": "not available"}, status=404)
        if request.method == "POST":
            if not self.cameras:
                return web.json_response({"error": "no camera is being watched"}, status=409)
            try:
                frames, every = int(request.query.get("frames", 240)), int(request.query.get("every", 15))
            except ValueError:
                return web.json_response({"error": "frames and every are numbers"}, status=400)
            if not bench.start(frames, every):
                return web.json_response({"error": "one is running already"}, status=409)
        return web.json_response(bench.status())

    async def _shutdown(self, _app) -> None:
        await self._stop_all()

    async def _stop_all(self) -> None:
        cameras, self.cameras = list(self.cameras.values()), {}
        for camera in cameras:
            await camera.stop()

    def status(self) -> dict[str, Any]:
        return {
            "type": "status", "mode": "tracking", "device": self.engine.device, "model": self.engine.model,
            "faces": self.engine.face is not None, "cameras": {cam: c.state() for cam, c in self.cameras.items()},
        }

    def _emit(self, camera: str, out: asyncio.Queue, packet: dict[str, Any], mode: str) -> None:
        """Called from a camera's thread."""
        message = json.dumps({"type": "tracks", "camera": camera, "mode": mode, "packet": packet})
        self._loop.call_soon_threadsafe(out.put_nowait, message)

    async def link(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
        await ws.prepare(request)
        await self._stop_all()  # one Home Assistant at a time: a new connection starts afresh
        self._loop = asyncio.get_running_loop()
        out = self._out = asyncio.Queue()
        log.info("Home Assistant connected")

        async def sender() -> None:
            while True:
                await ws.send_str(await out.get())

        async def say_status() -> None:
            while True:
                out.put_nowait(json.dumps(self.status()))
                await asyncio.sleep(STATUS_EVERY)

        tasks = [asyncio.create_task(sender()), asyncio.create_task(say_status())]
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                try:
                    data = json.loads(msg.data)
                    await self._message(data, out)
                except Exception:  # noqa: BLE001 - one bad message is no reason to hang up
                    log.debug("A message couldn't be used", exc_info=True)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._out is out:
                await self._stop_all()
            log.info("Home Assistant disconnected")
        return ws

    async def _message(self, data: dict[str, Any], out: asyncio.Queue) -> None:
        kind, cam = data.get("type"), data.get("camera")
        if not isinstance(cam, str):
            return
        if kind == "report":
            camera = self.cameras.get(cam)
            if camera is not None and isinstance(data.get("packet"), dict):
                camera.report(data["packet"])
        elif kind == "watch" and isinstance(data.get("source"), str):
            old = self.cameras.pop(cam, None)
            if old is not None:
                await old.stop()
            stamp = data.get("stamp")
            stamp = float(stamp) if isinstance(stamp, (int, float)) else DEFAULT_STAMP
            self.cameras[cam] = Camera(self.engine, data["source"], stamp, lambda packet, mode, cam=cam: self._emit(cam, out, packet, mode))
            log.info("Watching a camera (%d now)", len(self.cameras))
        elif kind == "sleep":
            old = self.cameras.pop(cam, None)
            if old is not None:
                await old.stop()
                log.info("A camera sleeps (%d watched)", len(self.cameras))
