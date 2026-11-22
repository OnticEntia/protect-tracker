"""The integration as Home Assistant runs it, against a stand-in service and
a stand-in for Protect Detections' relay."""

import asyncio
from unittest.mock import patch

from aiohttp import WSMsgType, web
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.protect_tracker.const import DOMAIN

WALKING = {"clockWall": 1, "clockStream": 1, "tracks": [{"id": "2", "objectType": "person", "stationary": False, "coord": [1, 2, 3, 4]}]}
PARKED = {"clockWall": 1, "clockStream": 1, "tracks": [{"id": "1", "objectType": "vehicle", "stationary": True, "coord": [1, 2, 3, 4]}]}
SOURCE = "rtsps://console.invalid:7441/KEY?enableSrtp"


class Relay:
    """Stands in for Protect Detections' hub."""

    def __init__(self):
        self.listeners = {}
        self.live = self

    def smart_cameras(self):
        return ["cam1"]

    def subscribe(self, camera, send):
        self.listeners[camera] = send
        return lambda: self.listeners.pop(camera, None)


class Service:
    """Stands in for the tracking service: keeps what it is sent."""

    def __init__(self):
        self.got = []
        self.connected = asyncio.Event()
        self.ws = None

    async def start(self, socket_enabled_port):
        app = web.Application()
        app.router.add_get("/link", self.link)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", socket_enabled_port)
        await site.start()

    async def link(self, request):
        ws = self.ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "status", "mode": "test", "device": "CPU"})
        self.connected.set()
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                self.got.append(msg.json())
        return ws

    async def wait_for(self, kind, n=1):
        for _ in range(200):
            if sum(1 for m in self.got if m["type"] == kind) >= n:
                return [m for m in self.got if m["type"] == kind]
            await asyncio.sleep(0.02)
        raise AssertionError(f"no {kind} came; got {[m['type'] for m in self.got]}")


async def _setup(hass, port, relay=None):
    hass.config.components.update({"camera", "unifiprotect"})  # stand-ins: nothing of them is used here
    if relay is not None:
        hass.data["protect_detections"] = relay
    entry = MockConfigEntry(domain=DOMAIN, options={"url": f"ws://127.0.0.1:{port}/link", "stream": "medium"})
    entry.add_to_hass(hass)
    with patch("custom_components.protect_tracker.async_get_stream_source", return_value=SOURCE), patch(
        "custom_components.protect_tracker.stream_entity", return_value=("camera.somewhere", 1)
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        yield entry


async def test_wakes_the_service_and_passes_reports_on(hass, socket_enabled, unused_tcp_port, hass_ws_client):
    service, relay = Service(), Relay()
    await service.start(unused_tcp_port)
    async for entry in _setup(hass, unused_tcp_port, relay):
        await asyncio.wait_for(service.connected.wait(), 5)
        hub = hass.data[DOMAIN]
        for _ in range(100):
            if hub.link.connected:
                break
            await asyncio.sleep(0.02)
        send = relay.listeners["cam1"]

        send({"status": "connected"})
        send(PARKED)  # a parked car alone wakes nothing
        await asyncio.sleep(0.1)
        assert service.got == []

        send(WALKING)
        watch = (await service.wait_for("watch"))[0]
        assert watch == {"type": "watch", "camera": "cam1", "source": SOURCE, "channel": 1}
        send(PARKED)  # once awake everything is passed on
        reports = await service.wait_for("report", 2)
        assert reports[0]["packet"]["tracks"][0]["id"] == "2" and reports[1]["packet"]["tracks"][0]["id"] == "1"

        client = await hass_ws_client(hass)
        await client.send_json({"id": 1, "type": "protect_tracker/status"})
        status = (await client.receive_json())["result"]
        assert status["service"]["connected"] and status["service"]["mode"] == "test"
        assert status["reports"] and status["cameras"] == ["cam1"] and status["watching"]["cam1"]["channel"] == 1
        assert "KEY" not in str(status)  # the stream address is not something to show

        hub.waker.idle = 0  # nothing moves any more
        await asyncio.sleep(0.01)
        hub._check()
        await service.wait_for("sleep")
        assert hub.watching == {}

        assert await hass.config_entries.async_unload(entry.entry_id)
        assert relay.listeners == {}
    await service.runner.cleanup()


async def test_waits_for_the_service_and_for_protect_detections(hass, socket_enabled, unused_tcp_port):
    async for entry in _setup(hass, unused_tcp_port):  # neither is there yet
        hub = hass.data[DOMAIN]
        assert hub.status()["service"]["connected"] is False and hub.status()["reports"] is False
        relay = Relay()
        hass.data["protect_detections"] = relay
        hub._check()
        assert "cam1" in relay.listeners
        relay.listeners["cam1"](WALKING)  # woken while the service is away
        service = Service()
        await service.start(unused_tcp_port)
        await asyncio.wait_for(service.connected.wait(), 10)
        for _ in range(100):
            if hub.link.connected:
                break
            await asyncio.sleep(0.02)
        hub._check()
        assert (await service.wait_for("watch"))[0]["camera"] == "cam1"
        await hass.config_entries.async_unload(entry.entry_id)
        await service.runner.cleanup()


async def test_config_flow(hass):
    hass.config.components.update({"camera", "unifiprotect"})
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    assert result["type"] == "form"
    with patch("custom_components.protect_tracker.async_setup_entry", return_value=True):
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {"url": "ws://127.0.0.1:8787/link", "stream": "high"})
    assert result["type"] == "create_entry" and result["options"]["stream"] == "high"
