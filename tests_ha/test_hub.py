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
        assert watch == {"type": "watch", "camera": "cam1", "source": SOURCE, "channel": 1, "stamp": -115.0}
        send(PARKED)  # once awake everything is passed on
        reports = await service.wait_for("report", 2)
        assert reports[0]["packet"]["tracks"][0]["id"] == "2" and reports[1]["packet"]["tracks"][0]["id"] == "1"

        client = await hass_ws_client(hass)
        await client.send_json({"id": 1, "type": "protect_tracker/status"})
        status = (await client.receive_json())["result"]
        assert status["service"]["connected"] and status["service"]["mode"] == "test"
        assert status["reports"] and status["cameras"] == ["cam1"] and status["watching"]["cam1"]["channel"] == 1
        assert "KEY" not in str(status)  # the stream address is not something to show

        # a card listening: Protect's boxes stamped with their frame's time, and the service's mixed in
        with patch("custom_components.protect_tracker.camera_of", return_value="cam1"):
            await client.send_json({"id": 2, "type": "protect_tracker/live/subscribe", "entity": "camera.somewhere"})
            assert (await client.receive_json())["success"]
        assert (await client.receive_json())["event"] == {"status": "connected", "mode": "auto", "service": True}
        send({**WALKING, "clockWall": 1000.0, "clockStream": 500.0, "clockStreamRate": 1000})
        theirs = (await client.receive_json())["event"]
        assert theirs["clockWall"] == 1000.0 and theirs["clockStream"] == 500.0  # as Protect sent it
        assert theirs["source"] == "local"  # the service is watching this camera: that is how it is served
        assert theirs["tracks"] == [{**WALKING["tracks"][0], "local": False}] and theirs["mode"] == "auto"  # no box of the service's yet: Protect's
        ours = {"source": "local", "clockWall": 1100.0, "clockStream": 400.0, "clockStreamRate": 1000,
                "tracks": [{"id": "2", "coord": [5, 6, 7, 8], "objectType": "person", "local": True}]}
        await service.ws.send_json({"type": "tracks", "camera": "cam1", "packet": ours})
        assert (await client.receive_json())["event"] == {**ours, "mode": "auto"}
        # one box a track: while the service's is fresh, Protect's for that id is held back (others pass)
        send({**WALKING, "tracks": WALKING["tracks"] + PARKED["tracks"]})
        both = (await client.receive_json())["event"]
        assert [tr["id"] for tr in both["tracks"]] == ["1"] and both["tracks"][0]["local"] is False and both["source"] == "local"
        hub._fresh["cam1"]["2"] -= 1.0  # the service has had no box of it for a moment
        send(WALKING)
        assert [tr["id"] for tr in (await client.receive_json())["event"]["tracks"]] == ["2"]
        send({"status": "disconnected"})
        assert (await client.receive_json())["event"] == {"status": "disconnected"}

        # on the service's own word: its own stream, nothing of Protect's passed on
        with patch("custom_components.protect_tracker.camera_of", return_value="cam1"):
            await client.send_json({"id": 7, "type": "protect_tracker/live/subscribe", "entity": "camera.somewhere", "mode": "local"})
            assert (await client.receive_json())["success"]
        assert (await client.receive_json())["event"] == {"status": "connected", "mode": "local", "service": True}
        recorded = []
        relay.record_live = lambda camera, packet: recorded.append((camera, packet["tracks"][0]["id"]))
        mine = {"source": "local", "mode": "local", "clockWall": 1200.0, "clockStream": 600.0, "clockStreamRate": 1000,
                "tracks": [{"id": "Lx-1", "coord": [1, 2, 3, 4], "objectType": "person", "local": True, "confirmed": False}]}
        await service.ws.send_json({"type": "tracks", "camera": "cam1", "mode": "local", "packet": mine})
        assert (await client.receive_json())["event"] == mine
        send(WALKING)  # Protect's report goes to the default stream's listener only
        got = await client.receive_json()
        assert got["id"] == 2 and got["event"]["source"] == "local" and got["event"]["tracks"][0]["local"] is False
        assert hub.seen["cam1"] == {"at": hub.seen["cam1"]["at"], "count": 1, "confirmed": False}  # a person, by the service's word
        assert recorded == [("cam1", "Lx-1")]  # and handed to the record of sightings
        await client.send_json({"id": 8, "type": "unsubscribe_events", "subscription": 7})
        assert (await client.receive_json())["success"]
        await client.send_json({"id": 9, "type": "protect_tracker/live/subscribe", "entity": "camera.somewhere", "mode": "nonsense"})
        assert (await client.receive_json())["success"] is False
        await client.send_json({"id": 13, "type": "protect_tracker/live/subscribe", "entity": "camera.unknown"})
        assert (await client.receive_json())["error"]["code"] == "not_found"
        await client.send_json({"id": 14, "type": "unsubscribe_events", "subscription": 2})
        assert (await client.receive_json())["success"]
        assert hub.listeners == {}

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
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {"url": "ws://127.0.0.1:8791/link", "stream": "high", "always": True})
    assert result["type"] == "create_entry" and result["options"]["stream"] == "high" and result["options"]["always"] is True


async def test_always_on_keeps_every_camera_watched(hass, socket_enabled, unused_tcp_port):
    service, relay = Service(), Relay()
    await service.start(unused_tcp_port)
    hass.config.components.update({"camera", "unifiprotect"})
    hass.data["protect_detections"] = relay
    entry = MockConfigEntry(domain=DOMAIN, options={"url": f"ws://127.0.0.1:{unused_tcp_port}/link", "stream": "medium", "always": True})
    entry.add_to_hass(hass)
    with patch("custom_components.protect_tracker.async_get_stream_source", return_value=SOURCE), patch(
        "custom_components.protect_tracker.stream_entity", return_value=("camera.somewhere", 1)
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await asyncio.wait_for(service.connected.wait(), 5)
        hub = hass.data[DOMAIN]
        for _ in range(100):
            if hub.link.connected:
                break
            await asyncio.sleep(0.02)
        hub._check()  # nothing has been reported, and still it is watched
        assert (await service.wait_for("watch"))[0]["camera"] == "cam1"
        relay.listeners["cam1"](PARKED)  # and everything Protect says is passed on
        await service.wait_for("report")
        hub.waker.idle = 0
        hub._check()
        await asyncio.sleep(0.1)
        assert not [m for m in service.got if m["type"] == "sleep"]
        await hass.config_entries.async_unload(entry.entry_id)
    await service.runner.cleanup()


class _Flags:
    smart_detect_types = ["person"]


class _Cam:
    id, mac, name, feature_flags = "cam1", "AA:BB:CC:00:11:22", "Somewhere", _Flags()


class _Api:
    class bootstrap:  # noqa: N801
        cameras = {"cam1": _Cam()}


async def test_person_seen_sensor(hass, socket_enabled, unused_tcp_port):
    service = Service()
    await service.start(unused_tcp_port)
    hass.config.components.update({"camera", "unifiprotect"})
    entry = MockConfigEntry(domain=DOMAIN, options={"url": f"ws://127.0.0.1:{unused_tcp_port}/link", "stream": "medium", "always": True})
    entry.add_to_hass(hass)
    with patch("custom_components.protect_tracker._protect_api", return_value=_Api()), patch(
        "custom_components.protect_tracker.binary_sensor._protect_api", return_value=_Api()
    ), patch("custom_components.protect_tracker.async_get_stream_source", return_value=SOURCE), patch(
        "custom_components.protect_tracker.stream_entity", return_value=("camera.somewhere", 1)
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await asyncio.wait_for(service.connected.wait(), 5)
        hub = hass.data[DOMAIN]
        for _ in range(100):
            if hub.link.connected:
                break
            await asyncio.sleep(0.02)
        hub._check()
        # with no Protect Detections at all the camera is watched all the same (always on)
        assert (await service.wait_for("watch"))[0]["camera"] == "cam1"
        entity = next(e for e in hass.states.async_entity_ids("binary_sensor") if "person_seen" in e)
        hub._seen_changed("cam1")
        await hass.async_block_till_done()
        assert hass.states.get(entity).state == "off"
        packet = {"source": "local", "clockWall": 1.0, "clockStream": 1.0, "tracks": [{"id": "Lx-1", "objectType": "person", "confirmed": True, "coord": [1, 2, 3, 4]}]}
        await service.ws.send_json({"type": "tracks", "camera": "cam1", "mode": "local", "packet": packet})
        for _ in range(100):
            if hass.states.get(entity).state == "on":
                break
            await asyncio.sleep(0.02)
        state = hass.states.get(entity)
        assert state.state == "on" and state.attributes["count"] == 1 and state.attributes["confirmed"] is True
        hub.seen["cam1"]["at"] -= 10  # nobody for a while
        hub._check()
        await hass.async_block_till_done()
        assert hass.states.get(entity).state == "off"
        await hass.config_entries.async_unload(entry.entry_id)
    await service.runner.cleanup()
