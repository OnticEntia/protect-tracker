"""Protect Tracker: what UniFi Protect reports, followed frame by frame.

UniFi Protect stays the authority on what is there: its live reports say that
something is being tracked on a camera, what it is and roughly where. A
tracking service on this machine (a container with the models; see the
README) then follows it in the camera's video on every frame.

This integration is the go-between, and holds no model itself:

  - it listens to Protect's live reports through the Protect Detections
    companion (which reads them with the UniFi Protect integration's own
    login: nothing is logged in separately and nothing is written to Protect);
  - when a camera reports something on the move it tells the service to watch
    that camera, handing it the camera's stream address as the UniFi Protect
    integration gives it (kept in memory only, never logged or stored);
  - it passes every report on to the service while the camera is watched, and
    tells it to stop when nothing has moved for a while.

UniFi Protect cameras only.

  websocket  protect_tracker/status           is the service reached, what is watched
  websocket  protect_tracker/live/subscribe   where things are in a camera's picture, live

`live/subscribe` gives what Protect Detections' `live/subscribe` gives, in the
same shape, with the service's boxes mixed in: for a thing the service follows
there is a box for every frame, under Protect's own track id and with
Protect's details; for everything else, and whenever the service is away or
has lost the thing, Protect's own boxes. The service's boxes are stamped as
Protect stamps its own, so a card times both the same way.

With `mode: "local"` the command gives the other stream instead: what the
service sees on its own word, under its own track ids, whether or not Protect
has reported anything; Protect's details are on a track once Protect has
reported the same thing (`protectId`, `confirmed`). Nothing of Protect's own
is passed on there. The README has the fields.

  binary sensor  "Person seen" per camera: on while the service itself sees a
                 person there (before, and whether or not, Protect says so)

In the default mode a track has one box at a time: while the service's box of it is fresh,
Protect's box for that id is left out of what is passed on, and comes back
only when the service has had none for a moment (HOLD_S). Each track says
which it is (`local`: true or false). `source` on a packet says how the
camera is being served just then, and stays put: "local" while the service
is watching it, "protect" while it is not.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from functools import partial
import logging
import time
from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.components.camera import async_get_stream_source
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import async_track_time_interval

from .const import CONF_ALWAYS, CONF_STREAM, CONF_URL, DEFAULT_STREAM, DEFAULT_URL, DETECTIONS, DOMAIN, STAMP_MS, STREAMS
from .link import ServiceLink
from .wake import Waker

_LOGGER = logging.getLogger(__name__)

CHECK = timedelta(seconds=2)
PLATFORMS = ["binary_sensor"]
SEEN_HOLD_S = 3.0  # "person seen" stays on this long after the service's last box of a person
HOLD_S = 0.7  # Protect's box of a thing is held back this long after the service's last box of it


def _protect_api(hass: HomeAssistant):
    """The UniFi Protect integration's logged-in client, or None if it isn't loaded."""
    for entry in hass.config_entries.async_entries("unifiprotect"):
        if entry.state is ConfigEntryState.LOADED and getattr(entry, "runtime_data", None):
            return entry.runtime_data.api
    return None


def stream_entity(hass: HomeAssistant, api: Any, camera_id: str, channel: int) -> tuple[str, int] | None:
    """The camera entity of one of a Protect camera's streams: the wanted
    channel if its entity is enabled, else whichever is. (entity id, channel)"""
    cam = api.bootstrap.cameras.get(camera_id) if api is not None else None
    if cam is None:
        return None
    mac = (cam.mac or "").replace(":", "").upper()
    reg = er.async_get(hass)
    by_uid = {
        e.unique_id: e.entity_id
        for e in reg.entities.values()
        if e.platform == "unifiprotect" and e.domain == "camera" and not e.disabled_by
    }
    for ch in (channel, 1, 0, 2):
        entity = by_uid.get(f"{mac}_{ch}")
        if entity:
            return entity, ch
    return None


def smart_cameras(api: Any) -> list[str]:
    """The cameras (Protect ids) that recognise things."""
    out = []
    for cam in api.bootstrap.cameras.values() if api is not None else []:
        try:
            if cam.feature_flags.smart_detect_types:
                out.append(cam.id)
        except AttributeError:
            pass
    return out


def camera_of(hass: HomeAssistant, api: Any, entity_id: str) -> str | None:
    """The Protect camera (its id) a camera entity belongs to."""
    entry = er.async_get(hass).async_get(entity_id)
    if entry is None or entry.platform != "unifiprotect" or entry.domain != "camera" or api is None:
        return None
    mac = str(entry.unique_id).rsplit("_", 1)[0].upper()
    for cam in api.bootstrap.cameras.values():
        if (cam.mac or "").replace(":", "").upper() == mac:
            return cam.id
    return None


Send = Callable[[dict[str, Any]], None]


class TrackerHub:
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.channel = STREAMS.get(entry.options.get(CONF_STREAM, DEFAULT_STREAM), 1)
        self.link = ServiceLink(hass, entry.options.get(CONF_URL, DEFAULT_URL), self._from_service, self._linked)
        self.waker = Waker()
        self.service: dict[str, Any] = {}  # what the service last said of itself
        self.watching: dict[str, dict[str, Any]] = {}  # camera -> {entity, channel, since}
        self.stamp = STAMP_MS.get(self.channel, -125.0)
        self.always = bool(entry.options.get(CONF_ALWAYS, False))
        self.listeners: dict[tuple[str, str], set[Send]] = {}  # (camera, mode) -> the cards listening to it
        self.seen: dict[str, dict[str, Any]] = {}  # camera -> the people the service sees there now: {at, count, confirmed}
        self.seen_listeners: dict[str, set[Callable[[], None]]] = {}
        self._was_linked = False
        self._fresh: dict[str, dict[str, float]] = {}  # camera -> track id -> when the service last sent a box of it
        self._relay: Any = None  # Protect Detections' hub the reports come from
        self._unsubs: dict[str, Callable[[], None]] = {}
        self._stop_check: Callable[[], None] | None = None

    async def start(self) -> None:
        self.link.start()
        self._attach()
        self._stop_check = async_track_time_interval(self.hass, self._check, CHECK)

    async def stop(self) -> None:
        if self._stop_check:
            self._stop_check()
        self._detach()
        await self.link.stop()

    # ------------------------------------------------- Protect's reports

    def _attach(self) -> None:
        """Listen to the live reports of every camera that recognises things.
        (Protect Detections may load after this, or be reloaded: checked again
        every few seconds.)"""
        relay = self.hass.data.get(DETECTIONS)
        if relay is self._relay and (relay is None or self._unsubs):
            return
        self._detach()
        self._relay = relay
        if relay is None:
            return
        try:
            cameras = list(relay.smart_cameras())
            for cam in cameras:
                self._unsubs[cam] = relay.live.subscribe(cam, partial(self._report, cam))
        except Exception:  # noqa: BLE001 - a Protect Detections without this: said once, tried again later
            _LOGGER.warning("Couldn't listen to Protect's live reports through Protect Detections", exc_info=True)
            self._detach()

    def _detach(self) -> None:
        for unsub in self._unsubs.values():
            try:
                unsub()
            except Exception:  # noqa: BLE001
                pass
        self._unsubs.clear()
        self._relay = None

    @callback
    def _report(self, camera: str, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("status"):
            self._tell(camera, payload)  # (of Protect's feed: nothing the local stream hangs on)
            return
        if (camera, "auto") in self.listeners:
            self._tell(camera, self._theirs(camera, payload))
        if self.waker.report(camera, payload, time.monotonic()):
            self.hass.async_create_task(self._watch(camera))
        if camera in self.waker.awake:
            self.link.send({"type": "report", "camera": camera, "packet": payload})

    async def _watch(self, camera: str) -> None:
        """Tell the service to watch a camera."""
        if not self.link.connected:
            return
        found = stream_entity(self.hass, _protect_api(self.hass), camera, self.channel)
        if found is None:
            return
        entity, channel = found
        try:
            source = await async_get_stream_source(self.hass, entity)
        except Exception:  # noqa: BLE001
            source = None
        if not source:
            _LOGGER.debug("No stream address for a camera that reported something")
            return
        if camera not in self.waker.awake:
            return  # gone quiet meanwhile
        if self.link.send({"type": "watch", "camera": camera, "source": source, "channel": channel, "stamp": STAMP_MS.get(channel, self.stamp)}):
            self.watching[camera] = {"entity": entity, "channel": channel, "since": time.time()}

    @callback
    def _check(self, _now=None) -> None:
        self._attach()
        if self.link.connected != self._was_linked:
            self._was_linked = self.link.connected
            if not self.link.connected:  # the service has gone: said to those who rely on it alone
                for camera, mode in list(self.listeners):
                    if mode == "local":
                        self._tell(camera, {"status": "disconnected", "mode": "local", "service": False}, "local")
        if self.always:  # every camera stays awake, whether or not Protect's reports are coming
            for camera in set(self._unsubs) | set(smart_cameras(_protect_api(self.hass))):
                self.waker.awake[camera] = time.monotonic()
        for camera in [cam for cam, info in self.seen.items() if time.monotonic() - info["at"] > SEEN_HOLD_S]:
            del self.seen[camera]  # nobody there any more
            self._seen_changed(camera)
        for camera in [] if self.always else self.waker.sweep(time.monotonic()):
            self.watching.pop(camera, None)
            self.link.send({"type": "sleep", "camera": camera})
        # awake but not watched (the service was away when it woke)
        if self.link.connected:
            for camera in self.waker.awake:
                if camera not in self.watching:
                    self.hass.async_create_task(self._watch(camera))

    # ------------------------------------------------------ the service

    @callback
    def _linked(self) -> None:
        self._fresh.clear()
        self.watching.clear()
        for camera, mode in list(self.listeners):
            if mode == "local":  # the service is back
                self._tell(camera, {"status": "connected", "mode": "local", "service": True}, "local")  # a service just (re)connected watches nothing yet: _check sees to it

    @callback
    def _from_service(self, message: dict[str, Any]) -> None:
        if message.get("type") == "status":
            self.service = {k: v for k, v in message.items() if k != "type"}
        elif message.get("type") == "tracks" and isinstance(message.get("packet"), dict):
            camera, now = str(message.get("camera")), time.monotonic()
            if message.get("mode") == "local":  # on the service's own word
                self._tell(camera, message["packet"], "local")
                if message["packet"].get("tracks"):
                    self._record(camera, message["packet"])
                people = [t for t in message["packet"].get("tracks") or [] if isinstance(t, dict) and t.get("objectType") == "person"]
                if people:
                    was = camera in self.seen
                    self.seen[camera] = {"at": now, "count": len(people), "confirmed": any(t.get("confirmed") for t in people)}
                    if not was:
                        self._seen_changed(camera)
                return
            fresh = self._fresh.setdefault(camera, {})
            for tr in message["packet"].get("tracks") or []:
                if isinstance(tr, dict):
                    fresh[str(tr.get("id"))] = now
            self._tell(camera, message["packet"])

    def _theirs(self, camera: str, payload: dict[str, Any]) -> dict[str, Any]:
        """One of Protect's reports as it is passed on: without the things
        the service has a fresh box of, and saying how the camera is served."""
        now = time.monotonic()
        fresh = self._fresh.get(camera) or {}
        for tid in [tid for tid, at in fresh.items() if now - at > 10]:
            del fresh[tid]
        tracks = [
            {**tr, "local": False}
            for tr in payload.get("tracks") or []
            if isinstance(tr, dict) and now - fresh.get(str(tr.get("id")), -1e9) > HOLD_S
        ]
        served = self.link.connected and camera in self.watching
        return {**payload, "tracks": tracks, "source": "local" if served else "protect", "mode": "auto"}

    # --------------------------------------------------------- the cards

    def _record(self, camera: str, packet: dict[str, Any]) -> None:
        """Hand what the service sees to Protect Detections' record of
        sightings, where that has a way in (boxes, kinds and times only)."""
        record = getattr(self.hass.data.get(DETECTIONS), "record_live", None)
        if callable(record):
            try:
                record(camera, packet)
            except Exception:  # noqa: BLE001 - the record is theirs to keep right
                _LOGGER.debug("Protect Detections couldn't take a sighting", exc_info=True)

    def _seen_changed(self, camera: str) -> None:
        for tell in list(self.seen_listeners.get(camera) or ()):
            tell()

    def _tell(self, camera: str, payload: dict[str, Any], mode: str = "auto") -> None:
        for send in list(self.listeners.get((camera, mode)) or ()):
            try:
                send(payload)
            except Exception:  # noqa: BLE001 - one listener gone is no reason to stop
                _LOGGER.debug("A live listener failed", exc_info=True)

    @callback
    def subscribe(self, camera: str, send: Send, mode: str = "auto") -> Callable[[], None]:
        key = (camera, mode)
        self.listeners.setdefault(key, set()).add(send)

        @callback
        def unsubscribe() -> None:
            listeners = self.listeners.get(key)
            if listeners is not None:
                listeners.discard(send)
                if not listeners:
                    del self.listeners[key]

        return unsubscribe

    def status(self) -> dict[str, Any]:
        return {
            "service": {"connected": self.link.connected, "error": self.link.error, **self.service},
            "reports": self._relay is not None and bool(self._unsubs),
            "cameras": sorted(self._unsubs),
            "watching": {cam: dict(info) for cam, info in self.watching.items()},
            "listeners": {f"{cam} {mode}": len(sends) for (cam, mode), sends in self.listeners.items()},
            "seen": {cam: {k: v for k, v in info.items() if k != "at"} for cam, info in self.seen.items()},
            "always": self.always,
        }


@websocket_api.websocket_command({vol.Required("type"): f"{DOMAIN}/status"})
@callback
def ws_status(hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]) -> None:
    """Is the service reached, are Protect's reports coming, what is watched."""
    hub: TrackerHub | None = hass.data.get(DOMAIN)
    if hub is None:
        connection.send_error(msg["id"], "not_loaded", "Protect Tracker isn't loaded")
        return
    connection.send_result(msg["id"], hub.status())


@websocket_api.websocket_command({
    vol.Required("type"): f"{DOMAIN}/live/subscribe",
    vol.Required("entity"): str,
    vol.Optional("mode", default="auto"): vol.In(["auto", "local"]),
})
@callback
def ws_live_subscribe(hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]) -> None:
    """Where things are in a camera's picture, live: Protect's reports with
    the service's boxes mixed in (see the top of this file)."""
    hub: TrackerHub | None = hass.data.get(DOMAIN)
    if hub is None:
        connection.send_error(msg["id"], "not_loaded", "Protect Tracker isn't loaded")
        return
    camera = camera_of(hass, _protect_api(hass), msg["entity"])
    if camera is None:
        connection.send_error(msg["id"], "not_found", "Not a UniFi Protect camera")
        return
    mode = msg["mode"]
    if mode == "auto" and camera not in hub._unsubs:  # noqa: SLF001
        connection.send_error(msg["id"], "not_ready", "Protect's live reports for this camera aren't being received")
        return
    if mode == "local" and not hub.link.connected:
        connection.send_error(msg["id"], "not_ready", "The tracking service isn't running")
        return

    @callback
    def send(payload: dict[str, Any]) -> None:
        connection.send_message(websocket_api.event_message(msg["id"], payload))

    connection.send_result(msg["id"])
    connection.subscriptions[msg["id"]] = hub.subscribe(camera, send, mode)
    send({"status": "connected", "mode": mode, "service": hub.link.connected})


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    hub = TrackerHub(hass, entry)
    await hub.start()
    hass.data[DOMAIN] = hub
    if not hass.data.get(f"{DOMAIN}_registered"):
        websocket_api.async_register_command(hass, ws_status)
        websocket_api.async_register_command(hass, ws_live_subscribe)
        hass.data[f"{DOMAIN}_registered"] = True
    entry.async_on_unload(entry.add_update_listener(_reload))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def _reload(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    hub: TrackerHub | None = hass.data.pop(DOMAIN, None)
    if hub is not None:
        await hub.stop()
    return True
