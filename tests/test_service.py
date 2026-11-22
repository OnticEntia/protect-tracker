"""The service end to end: a made-up camera, a stand-in detector, and a
stand-in for Home Assistant on the link."""

import asyncio
import json
import time

import aiohttp
from aiohttp import web
import numpy as np

from fakes import FakeCamera, encode, packetize
from tracker_service.server import Service
from tracker_service.video import picture

FIRST = 90 * 50_000  # the first frame's RTP timestamp: the stream clock at 50 s


class SquareFinder:
    """Finds the white square of the made-up camera's pictures and calls it a person."""

    device, model, face = "test", "square", None

    def look(self, frame, far: bool = False) -> dict:
        img = picture(frame)
        cols = np.where(img[120, :, 1] > 128)[0]
        if not len(cols):
            return {"person": []}
        h, w = img.shape[:2]
        return {"person": [(cols[0] / w, 100 / h, (cols[-1] + 1) / w, 140 / h, 0.9)]}


def square(i: float) -> list:
    """Where the square is in frame i, as Protect would say it: [x, y, w, h] in 0-1000."""
    return [10 * i / 320 * 1000, 100 / 240 * 1000, 40 / 320 * 1000, 40 / 240 * 1000]


def test_boxes_for_every_frame_under_protects_id():
    async def go():
        service = Service(SquareFinder())
        runner = web.AppRunner(service.app())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        coded = encode("h265", 24)
        got, statuses, local = [], [], []
        async with FakeCamera("h265", packetize("h265", coded, FIRST, mtu=600), pace=1 / 30) as cam, aiohttp.ClientSession() as session:
            ws = await session.ws_connect(f"ws://127.0.0.1:{port}/link")
            wall0 = time.time() * 1000

            def report(i: int) -> dict:
                # what Protect says of frame i: stamped 125 ms after it
                ms = i * 1000 / 30 + 125
                return {"type": "report", "camera": "c1", "packet": {
                    "clockWall": wall0 + ms, "clockStream": 50_000 + ms, "clockStreamRate": 1000, "eventId": "e9",
                    "tracks": [{"id": "T7", "objectType": "person", "confidence": 77, "stationary": False, "zones": [2], "coord": square(i)}]}}

            await ws.send_json(report(0))  # before it is watched: dropped
            await ws.send_json({"type": "watch", "camera": "c1", "source": cam.url + "?enableSrtp", "channel": 1, "stamp": -125})
            await ws.send_json(report(0))
            await asyncio.sleep(1.0)  # the video plays; the reports of it come late, as they do
            for i in (6, 12, 18):
                await ws.send_json(report(i))
            end = time.monotonic() + 3
            while time.monotonic() < end:
                try:
                    msg = await asyncio.wait_for(ws.receive(), 0.5)
                except asyncio.TimeoutError:
                    continue
                data = json.loads(msg.data)
                if data["type"] == "tracks" and data["mode"] == "local":
                    local.append(data)
                else:
                    (got if data["type"] == "tracks" else statuses).append(data)
            state = service.status()
            await ws.send_json({"type": "sleep", "camera": "c1"})
            await asyncio.sleep(0.2)
            assert service.cameras == {}
            await ws.close()
        await runner.cleanup()
        return wall0, got, statuses, state, local

    wall0, got, statuses, state, local = asyncio.run(go())
    # on its own word the square was told well before Protect's report of it was bound, under an id of the service's own
    assert local and local[0]["packet"]["tracks"][0]["id"].startswith("L") and any(m["packet"]["tracks"][0].get("protectId") == "T7" for m in local)
    assert len(local) > len(got) and state["cameras"]["c1"]["local"]["told"] >= 1  # (the made-up video is played again when it ends)
    assert statuses and statuses[0]["device"] == "test"
    cam = state["cameras"]["c1"]
    assert cam["codec"] == "h265" and cam["bound"] == 1 and cam["frames"] >= 20
    assert "KEY" not in json.dumps(state) and "rtsp" not in json.dumps(state)
    frames = {}
    for message in got:
        assert message["camera"] == "c1"
        packet = message["packet"]
        (tr,) = packet["tracks"]
        assert tr["id"] == "T7" and tr["objectType"] == "person" and tr["zones"] == [2] and tr["local"] and packet["eventId"] == "e9"
        i = (packet["clockWall"] - 125 - wall0) * 30 / 1000  # which frame this box says it is of (stamped as Protect stamps)
        assert abs(i - round(i)) < 0.01
        frames[round(i)] = tr["coord"]
    assert len(frames) >= 17 and len(frames) == len(got)  # a box for every frame after the one it was bound at, each once
    assert min(frames) == 7
    for i, coord in frames.items():
        assert abs(coord[0] - square(i)[0]) < 12, f"frame {i}: the box is not where the square was in that frame"
