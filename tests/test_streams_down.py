"""A camera whose stream cannot be read: tried again less and less often,
said once when it is down and once when it is back, and dropped when told."""

import asyncio
import json
import socket

import aiohttp
from aiohttp import web

from fakes import FakeCamera, encode, packetize
from test_service import FIRST, SquareFinder
from tracker_service import camera as camera_module
from tracker_service.server import Service


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def serve(service: Service):
    runner = web.AppRunner(service.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


async def statuses(ws, until, timeout: float = 5.0) -> list[dict]:
    """The status messages as they come, until one that `until` likes."""
    got = []
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        try:
            msg = await asyncio.wait_for(ws.receive(), 0.2)
        except asyncio.TimeoutError:
            continue
        data = json.loads(msg.data)
        if data["type"] == "status":
            got.append(data)
            if until(data):
                return got
    raise AssertionError(f"it never came; the last: {got[-1] if got else None}")


def test_down_is_said_once_and_back_is_said(monkeypatch, caplog):
    monkeypatch.setattr(camera_module, "DOWN_AFTER", 0.3)
    monkeypatch.setattr(camera_module, "RETRY_FIRST", 0.05)
    monkeypatch.setattr(camera_module, "RETRY_MOST", 0.2)
    caplog.set_level("INFO", logger="tracker_service.camera")

    async def go():
        service = Service(SquareFinder())
        runner, port = await serve(service)
        cam_port = free_port()  # nothing listens there yet: the camera is away
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect(f"ws://127.0.0.1:{port}/link")
            await ws.send_json({"type": "watch", "camera": "c1", "source": f"rtsp://127.0.0.1:{cam_port}/SECRET", "channel": 1, "stamp": -125})
            first = await statuses(ws, lambda s: s["cameras"].get("c1", {}).get("stream") == "down")
            down = first[-1]["cameras"]["c1"]
            assert down["error"] and down["opened"] == 0 and down["retry_s"] <= 0.2 and down["down_s"] is not None
            assert down["below"] is False  # no frames is not "the detector cannot keep up"
            tries_then = service.cameras["c1"].retry
            await asyncio.sleep(0.6)  # it goes on trying, and says nothing more of it
            assert tries_then <= 0.2 and service.cameras["c1"].retry == 0.2  # the wait has grown to its most, and no further
            coded = encode("h265", 12)
            async with FakeCamera("h265", packetize("h265", coded, FIRST, mtu=600), pace=1 / 30, port=cam_port, path="/SECRET"):
                back = await statuses(ws, lambda s: s["cameras"]["c1"]["stream"] == "open")
                assert back[-1]["cameras"]["c1"]["error"] is None and back[-1]["cameras"]["c1"]["opened"] >= 1
            worker = service.cameras["c1"].worker
            await ws.send_json({"type": "sleep", "camera": "c1"})  # removed: told to stop
            await asyncio.sleep(0.3)
            assert service.cameras == {} and not worker.is_alive()  # nothing of it is left running
            assert "c1" not in service.status()["cameras"]
            await ws.close()
        await runner.cleanup()

    asyncio.run(go())
    said = [r.getMessage() for r in caplog.records if "stream" in r.getMessage()]
    assert len([m for m in said if "is down" in m]) == 1 and len([m for m in said if "is back" in m]) == 1  # a line per change, not per try
    assert "SECRET" not in caplog.text and "127.0.0.1" not in caplog.text  # never the address


def test_a_short_drop_is_not_worth_a_word(monkeypatch, caplog):
    monkeypatch.setattr(camera_module, "RETRY_FIRST", 0.05)
    caplog.set_level("INFO", logger="tracker_service.camera")

    async def go():
        service = Service(SquareFinder())
        runner, port = await serve(service)
        coded = encode("h265", 6)
        async with FakeCamera("h265", packetize("h265", coded, FIRST, mtu=600), pace=1 / 30) as cam, aiohttp.ClientSession() as session:
            ws = await session.ws_connect(f"ws://127.0.0.1:{port}/link")
            await ws.send_json({"type": "watch", "camera": "c1", "source": cam.url, "channel": 1, "stamp": -125})
            await asyncio.sleep(1.5)  # the made-up stream ends every fifth of a second and is opened again
            state = service.status()["cameras"]["c1"]
            assert state["opened"] >= 3 and state["stream"] != "down"
            await ws.close()
        await runner.cleanup()

    asyncio.run(go())
    assert not [r for r in caplog.records if "is down" in r.getMessage()]


def test_a_connection_cut_in_the_middle_is_tried_again(monkeypatch):
    monkeypatch.setattr(camera_module, "RETRY_FIRST", 0.05)
    opened = []

    class Cut:
        """A stream that opens and is then cut off, with an error that names the address."""

        def __init__(self, source: str) -> None:
            opened.append(source)

        async def __aenter__(self):
            from tracker_service.rtsp import VideoTrack

            self.track = VideoTrack("h265", 96, "")
            return self

        async def __aexit__(self, *exc) -> None:
            return None

        async def packets(self):
            raise ConnectionResetError("rtsp://somewhere/KEY was reset")
            yield  # pragma: no cover

    monkeypatch.setattr(camera_module, "RtspReader", Cut)

    async def go():
        cam = camera_module.Camera(SquareFinder(), "rtsp://somewhere/KEY", -125, lambda packet, mode: None)
        await asyncio.sleep(0.5)
        state = cam.state()
        await cam.stop()
        return state

    state = asyncio.run(go())
    assert len(opened) >= 3  # it did not give up at the first cut
    assert state["error"] == "ConnectionResetError" and "KEY" not in json.dumps(state)
