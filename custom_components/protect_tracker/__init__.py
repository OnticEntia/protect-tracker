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

  websocket  protect_tracker/status   is the service reached, what is watched
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

from .const import CONF_STREAM, CONF_URL, DEFAULT_STREAM, DEFAULT_URL, DETECTIONS, DOMAIN, STREAMS
from .link import ServiceLink
from .wake import Waker

_LOGGER = logging.getLogger(__name__)

CHECK = timedelta(seconds=5)


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


class TrackerHub:
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.channel = STREAMS.get(entry.options.get(CONF_STREAM, DEFAULT_STREAM), 1)
        self.link = ServiceLink(hass, entry.options.get(CONF_URL, DEFAULT_URL), self._from_service, self._linked)
        self.waker = Waker()
        self.service: dict[str, Any] = {}  # what the service last said of itself
        self.watching: dict[str, dict[str, Any]] = {}  # camera -> {entity, channel, since}
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
        if not isinstance(payload, dict) or payload.get("status"):
            return
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
        if self.link.send({"type": "watch", "camera": camera, "source": source, "channel": channel}):
            self.watching[camera] = {"entity": entity, "channel": channel, "since": time.time()}

    @callback
    def _check(self, _now=None) -> None:
        self._attach()
        for camera in self.waker.sweep(time.monotonic()):
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
        self.watching.clear()  # a service just (re)connected watches nothing yet: _check sees to it

    @callback
    def _from_service(self, message: dict[str, Any]) -> None:
        if message.get("type") == "status":
            self.service = {k: v for k, v in message.items() if k != "type"}

    def status(self) -> dict[str, Any]:
        return {
            "service": {"connected": self.link.connected, "error": self.link.error, **self.service},
            "reports": self._relay is not None and bool(self._unsubs),
            "cameras": sorted(self._unsubs),
            "watching": {cam: dict(info) for cam, info in self.watching.items()},
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


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    hub = TrackerHub(hass, entry)
    await hub.start()
    hass.data[DOMAIN] = hub
    if not hass.data.get(f"{DOMAIN}_registered"):
        websocket_api.async_register_command(hass, ws_status)
        hass.data[f"{DOMAIN}_registered"] = True
    entry.async_on_unload(entry.add_update_listener(_reload))
    return True


async def _reload(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    hub: TrackerHub | None = hass.data.pop(DOMAIN, None)
    if hub is not None:
        await hub.stop()
    return True
