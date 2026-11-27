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

Cameras come and go (roster.py). One that goes offline is left alone until it
is back: the service is told to stop, its sensor is unavailable, and whoever
listens is sent {"status": "offline", "reason": "offline" | "no_stream" |
"stream_down"} and, when it is back, {"status": "online"}. One that is taken
out of Protect is forgotten: the service is told to stop for good, listeners
are sent {"status": "removed", "error": {...}} and their subscriptions are
ended, and its sensor is removed. One that is adopted is taken up without a
restart.

Every message, status messages included, says whether the camera's whole view is moving
(`view`: "moving" | "steady"; a camera that pans, tilts or zooms, told by the
service from the picture). A change is also said at once, on its own:
{"status": "view", "view": ...}.
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
from .roster import NO_STREAM, OFFLINE, READY, Roster
from .wake import Waker

_LOGGER = logging.getLogger(__name__)

CHECK = timedelta(seconds=2)
PLATFORMS = ["binary_sensor"]
SEEN_HOLD_S = 3.0  # "person seen" stays on this long after the service's last box of a person
STREAM_FEATURE = 2  # (CameraEntityFeature.STREAM)
NO_ENTITY, NO_RTSP = "no_camera_entity", "rtsp_not_enabled"
LACKS = {NO_ENTITY: "none of its camera entities is enabled in Home Assistant",
         NO_RTSP: "its camera entity has no stream: enable RTSP for the camera in Protect, or use the UniFi Protect integration's repair for it"}
REASK_S = 30.0  # while a camera's stream is down its address is asked for again this often (it may have changed)
HOLD_S = 0.7  # Protect's box of a thing is held back this long after the service's last box of it


def _protect_api(hass: HomeAssistant):
    """The UniFi Protect integration's logged-in client, or None if it isn't loaded."""
    for entry in hass.config_entries.async_entries("unifiprotect"):
        if entry.state is ConfigEntryState.LOADED and getattr(entry, "runtime_data", None):
            return entry.runtime_data.api
    return None


def camera_entities(hass: HomeAssistant) -> dict[str, str]:
    """The UniFi Protect integration's camera entities that are enabled: unique id -> entity id."""
    return {
        e.unique_id: e.entity_id
        for e in er.async_get(hass).entities.values()
        if e.platform == "unifiprotect" and e.domain == "camera" and not e.disabled_by
    }


def can_stream(hass: HomeAssistant, entity_id: str) -> bool:
    """Does this camera entity have a stream just now? The UniFi Protect
    integration keeps a camera entity for a camera none of whose streams is
    shared (RTSP is not enabled for it in Protect): it shows still pictures
    only, and says so by not having the stream feature."""
    state = hass.states.get(entity_id)
    if state is None:
        return False
    try:
        return bool(int(state.attributes.get("supported_features") or 0) & STREAM_FEATURE)
    except (TypeError, ValueError):
        return False


def stream_entity(hass: HomeAssistant, api: Any, camera_id: str, channel: int, by_uid: dict[str, str] | None = None) -> tuple[str, int] | None:
    """The camera entity of one of a Protect camera's streams: the wanted
    channel if its entity is enabled and has a stream, else whichever is and
    has. (entity id, channel)"""
    cam = api.bootstrap.cameras.get(camera_id) if api is not None else None
    if cam is None:
        return None
    mac = (cam.mac or "").replace(":", "").upper()
    if by_uid is None:
        by_uid = camera_entities(hass)
    for ch in (channel, 1, 0, 2):
        entity = by_uid.get(f"{mac}_{ch}")
        if entity and can_stream(hass, entity):
            return entity, ch
    return None


def why_no_stream(hass: HomeAssistant, cam: Any, by_uid: dict[str, str] | None = None) -> str:
    """Why a camera has no stream to watch: none of its camera entities is
    enabled (NO_ENTITY), or one is, but Home Assistant has no stream for it:
    RTSP is not enabled for the camera in Protect, or the UniFi Protect
    integration has not heard of it yet (NO_RTSP)."""
    mac = (getattr(cam, "mac", None) or "").replace(":", "").upper()
    if by_uid is None:
        by_uid = camera_entities(hass)
    return NO_RTSP if any(uid.rsplit("_", 1)[0].upper() == mac for uid in by_uid) else NO_ENTITY


def _connected(cam: Any) -> bool:
    """Does Protect say this camera is connected just now?"""
    state = getattr(cam, "state", None)
    if state is not None:
        return str(getattr(state, "value", state)).upper() == "CONNECTED"
    return bool(getattr(cam, "is_connected", True))


def cameras_now(hass: HomeAssistant, api: Any, channel: int, streams: dict[str, tuple[str, int]] | None = None,
                why: dict[str, str] | None = None) -> dict[str, str] | None:
    """The cameras this Protect has that recognise things, and how each
    stands (roster.py). None when Protect cannot be asked (the UniFi Protect
    integration isn't loaded just now). `streams` is filled with the stream
    each ready camera is to be watched by: (entity id, channel); `why` with
    why each camera that has none has none."""
    if api is None:
        return None
    by_uid = camera_entities(hass)
    out: dict[str, str] = {}
    for cam in api.bootstrap.cameras.values():
        try:
            smart = bool(cam.feature_flags.smart_detect_types)
        except AttributeError:
            smart = False
        # (a camera this console can see but has not adopted is not one of its cameras)
        if not smart or not getattr(cam, "mac", None) or not getattr(cam, "is_adopted_by_us", True):
            continue
        if not _connected(cam):
            out[cam.id] = OFFLINE
        elif (found := stream_entity(hass, api, cam.id, channel, by_uid)) is None:
            out[cam.id] = NO_STREAM
            if why is not None:
                why[cam.id] = why_no_stream(hass, cam, by_uid)
        else:
            out[cam.id] = READY
            if streams is not None:
                streams[cam.id] = found
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
        self.roster = Roster()  # which cameras there are, and how each stands
        self.macs: dict[str, str] = {}  # camera -> its hardware address (what its device is known by)
        self.stream_down: set[str] = set()  # watched, but the service has had no video of it for a while
        self.view: dict[str, str] = {}  # camera -> "moving" while the service finds its whole view moving (a camera that turns)
        self.add_sensor: Callable[[str], None] | None = None  # (binary_sensor.py) makes a camera's sensor
        self.gone_listeners: dict[str, set[Callable[[], None]]] = {}  # camera -> told when it is removed for good
        self._ends: dict[Send, Callable[[], None]] = {}  # a listener -> what ends its subscription
        self.lacking: dict[str, str] = {}  # camera -> why it has no stream to watch (NO_ENTITY | NO_RTSP)
        self.no_address: set[str] = set()  # cameras whose camera entity says it has a stream but gives no address for it
        self._streams: dict[str, tuple[str, int]] = {}  # camera -> the stream it is to be watched by just now: (entity, channel)
        self._waking: set[str] = set()  # cameras whose stream address is being fetched just now
        self._relay_failed: Any = None  # the relay that could not be listened to (said once)
        self._was_linked = False
        self._fresh: dict[str, dict[str, float]] = {}  # camera -> track id -> when the service last sent a box of it
        self._relay: Any = None  # Protect Detections' hub the reports come from
        self._unsubs: dict[str, Callable[[], None]] = {}
        self._stop_check: Callable[[], None] | None = None

    async def start(self) -> None:
        self.link.start()
        self._cameras()
        self._stop_check = async_track_time_interval(self.hass, self._check, CHECK)

    async def stop(self) -> None:
        if self._stop_check:
            self._stop_check()
        self._detach()
        await self.link.stop()

    # ------------------------------------------------- Protect's reports

    def _cameras(self) -> None:
        """Keep up with which cameras there are (roster.py) and act on what
        has changed; listen to the live reports of those that can send any.
        (Protect Detections may load after this, or be reloaded; a camera may
        be adopted, go offline, come back, or be removed: checked every few
        seconds.)"""
        relay = self.hass.data.get(DETECTIONS)
        if relay is not self._relay:
            self._detach()
            self._relay = relay
        api = _protect_api(self.hass)
        streams: dict[str, tuple[str, int]] = {}
        lacking: dict[str, str] = {}
        seen = cameras_now(self.hass, api, self.channel, streams, lacking)
        if seen is not None:
            self._streams, self.lacking = streams, lacking
        for camera, was, now in self.roster.update(seen):
            if now is None:
                self._removed(camera)
            elif was is None:
                cam = api.bootstrap.cameras.get(camera) if api is not None else None
                self.macs[camera] = getattr(cam, "mac", None) or ""
                _LOGGER.info("A camera has been taken up (%d now)", len(self.roster.cameras))
                if now == NO_STREAM:
                    _LOGGER.info("It has no stream to watch yet (%s): looked for again every %d s", LACKS.get(lacking.get(camera), "not known why"), CHECK.total_seconds())
            elif now == READY:
                _LOGGER.info("A camera is back")
                self._say(camera, "online")
            else:
                _LOGGER.info("A camera is %s: it is left alone until it is back",
                             "offline" if now == OFFLINE else f"without a stream to watch ({LACKS.get(lacking.get(camera), 'not known why')})")
                self._let_go(camera)
                self._say(camera, "offline", now)
            self._seen_changed(camera)
        if self.add_sensor is not None:  # (a camera just adopted gets its sensor once its device is there: asked again until then)
            for camera in self.roster.cameras:
                self.add_sensor(camera)
        # an offline camera has nothing to report; listening would only be knocking at its door
        wanted = {cam for cam, state in self.roster.cameras.items() if state != OFFLINE} if relay is not None and relay is not self._relay_failed else set()
        for camera in set(self._unsubs) - wanted:
            self._unlisten(camera)
        try:
            for camera in wanted - set(self._unsubs):
                self._unsubs[camera] = relay.live.subscribe(camera, partial(self._report, camera))
        except Exception:  # noqa: BLE001 - a Protect Detections without this: said once, tried again when it is reloaded
            _LOGGER.warning("Couldn't listen to Protect's live reports through Protect Detections", exc_info=True)
            self._detach()
            self._relay, self._relay_failed = relay, relay

    def _unlisten(self, camera: str) -> None:
        unsub = self._unsubs.pop(camera, None)
        if unsub is not None:
            try:
                unsub()
            except Exception:  # noqa: BLE001
                pass

    def _let_go(self, camera: str) -> None:
        """Stop watching a camera, and drop what was held of it."""
        self.waker.forget(camera)
        self._fresh.pop(camera, None)
        self.stream_down.discard(camera)
        self.no_address.discard(camera)
        self.view.pop(camera, None)
        self.seen.pop(camera, None)
        self.watching.pop(camera, None)
        self.link.send({"type": "sleep", "camera": camera})  # (whether or not it was watched: the service may know better)

    def _removed(self, camera: str) -> None:
        """A camera is no longer in Protect: everything of it is dropped, and
        whoever listens is told so and let go."""
        _LOGGER.info("A camera is no longer in Protect: it is forgotten (%d left)", len(self.roster.cameras))
        self._let_go(camera)
        self._unlisten(camera)
        for mode in ("auto", "local"):
            for send in list(self.listeners.pop((camera, mode), ())):
                try:
                    send({"status": "removed", "mode": mode, "service": self.link.connected, "view": "steady",
                          "error": {"code": "camera_removed", "message": "This camera is no longer in UniFi Protect"}})
                except Exception:  # noqa: BLE001
                    pass
                end = self._ends.pop(send, None)
                if end is not None:
                    end()
        for gone in list(self.gone_listeners.pop(camera, ())):
            gone()
        self.seen_listeners.pop(camera, None)
        self.macs.pop(camera, None)

    def _view(self, camera: str, view: Any) -> None:
        """What the service says of a camera's view: kept (it goes on every
        packet of Protect's passed on), and a change is said at once."""
        view = "moving" if view == "moving" else "steady"
        if self.view.get(camera, "steady") != view:
            self.view[camera] = view
            for mode in ("auto", "local"):
                self._tell(camera, {"status": "view", "view": view, "mode": mode, "service": self.link.connected}, mode)

    def _say(self, camera: str, status: str, reason: str | None = None) -> None:
        """Tell whoever listens to a camera how it stands."""
        for mode in ("auto", "local"):
            payload = {"status": status, "mode": mode, "service": self.link.connected, "view": self.view.get(camera, "steady")}
            if reason:
                payload["reason"] = reason
            self._tell(camera, payload, mode)

    def state(self, camera: str) -> str | None:
        """How a camera stands: watching | asleep | offline | no_stream |
        stream_down; None for one that is not (or no longer) there."""
        state = self.roster.cameras.get(camera)
        if state != READY:
            return state
        if camera in self.stream_down:
            return "stream_down"
        if camera in self.no_address and camera not in self.watching:
            return NO_STREAM
        return "watching" if self.link.connected and camera in self.watching else "asleep"

    def why(self, camera: str) -> str | None:
        """Why a camera is not being watched (None: it is)."""
        state = self.state(camera)
        if state == "watching":
            return None
        if state == OFFLINE:
            return "offline"  # Protect says it is not connected (or Protect cannot be asked)
        if state == NO_STREAM:
            return "no_address" if self.roster.cameras.get(camera) == READY else self.lacking.get(camera, NO_ENTITY)
        if state == "stream_down":
            return "stream_down"  # the service cannot read its stream (its error is in the service's part of the status)
        if not self.link.connected:
            return "service_away"
        if camera not in self.waker.awake:
            return "quiet"  # nothing moving that Protect has reported: it is watched when something does
        return "starting"  # about to be: its stream's address is being fetched or has just been sent

    def _detach(self) -> None:
        for unsub in self._unsubs.values():
            try:
                unsub()
            except Exception:  # noqa: BLE001
                pass
        self._unsubs.clear()
        self._relay = self._relay_failed = None

    @callback
    def _report(self, camera: str, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("status"):
            self._tell(camera, {**payload, "view": self.view.get(camera, "steady")})  # (of Protect's feed: nothing the local stream hangs on)
            return
        if (camera, "auto") in self.listeners:
            self._tell(camera, self._theirs(camera, payload))
        if self.roster.cameras.get(camera) != READY:
            return  # no stream of it to be had: Protect's own boxes are all there is
        if self.waker.report(camera, payload, time.monotonic()):
            self.hass.async_create_task(self._watch(camera))
        if camera in self.waker.awake:
            self.link.send({"type": "report", "camera": camera, "packet": payload})

    async def _watch(self, camera: str, again: bool = False) -> None:
        """Tell the service to watch a camera. `again`: it is being watched,
        but its stream may be another by now (the stream wanted has become
        available, or the address has changed while the service could not
        read it): told anew only if so."""
        if not self.link.connected or camera in self._waking or self.roster.cameras.get(camera) != READY:
            return
        self._waking.add(camera)
        try:
            found = stream_entity(self.hass, _protect_api(self.hass), camera, self.channel)
            if found is None:
                return
            entity, channel = found
            try:
                source = await async_get_stream_source(self.hass, entity)
            except Exception:  # noqa: BLE001
                source = None
            if not source:
                if camera not in self.no_address:
                    self.no_address.add(camera)
                    _LOGGER.info("A camera's entity gives no stream address yet: asked again every %d s", CHECK.total_seconds())
                    self._seen_changed(camera)
                return
            if camera in self.no_address:
                self.no_address.discard(camera)
                _LOGGER.info("A camera's stream address is there now")
                self._seen_changed(camera)
            if camera not in self.waker.awake or self.roster.cameras.get(camera) != READY:
                return  # gone quiet, or gone, meanwhile
            key, held = hash(source), self.watching.get(camera)
            if again and held is not None and held["entity"] == entity and held["key"] == key:
                held["asked"] = time.monotonic()  # the same as before: the service goes on trying it
                return
            if self.link.send({"type": "watch", "camera": camera, "source": source, "channel": channel, "stamp": STAMP_MS.get(channel, self.stamp)}):
                # (of the address only a number to tell a change by is kept)
                self.watching[camera] = {"entity": entity, "channel": channel, "since": time.time(), "key": key, "asked": time.monotonic()}
        finally:
            self._waking.discard(camera)

    @callback
    def _check(self, _now=None) -> None:
        self._cameras()
        if self.link.connected != self._was_linked:
            self._was_linked = self.link.connected
            for camera in list(self.seen_listeners):  # the sensors are unavailable without the service, and back with it
                self._seen_changed(camera)
            if not self.link.connected:  # the service has gone: said to those who rely on it alone
                for camera, mode in list(self.listeners):
                    if mode == "local":
                        self._tell(camera, {"status": "disconnected", "mode": "local", "service": False, "view": "steady"}, "local")
        if self.always:  # every camera that can be watched stays awake, whether or not Protect's reports are coming
            for camera in self.roster.ready():
                self.waker.awake[camera] = time.monotonic()
        for camera in [cam for cam, info in self.seen.items() if time.monotonic() - info["at"] > SEEN_HOLD_S]:
            del self.seen[camera]  # nobody there any more
            self._seen_changed(camera)
        for camera in [] if self.always else self.waker.sweep(time.monotonic()):
            self.watching.pop(camera, None)
            self.link.send({"type": "sleep", "camera": camera})
        # awake but not watched (the service was away when it woke)
        if self.link.connected:
            for camera in list(self.waker.awake):
                if camera not in self.watching and camera not in self._waking:
                    self.hass.async_create_task(self._watch(camera))
            # watched, but by a stream that is no longer the one to use (a camera just adopted gets its streams
            # one after another), or by an address the service cannot read (it may have been given a new one)
            for camera, held in list(self.watching.items()):
                found = self._streams.get(camera)
                changed = found is not None and found[0] != held["entity"]
                if changed or (camera in self.stream_down and time.monotonic() - held["asked"] > REASK_S):
                    self.hass.async_create_task(self._watch(camera, again=True))

    # ------------------------------------------------------ the service

    @callback
    def _linked(self) -> None:
        self._fresh.clear()
        self.watching.clear()
        self.stream_down.clear()
        self.view.clear()
        for camera, mode in list(self.listeners):
            if mode == "local":  # the service is back
                self._tell(camera, {"status": "connected", "mode": "local", "service": True, "view": "steady"}, "local")  # a service just (re)connected watches nothing yet: _check sees to it

    @callback
    def _from_service(self, message: dict[str, Any]) -> None:
        if message.get("type") == "status":
            self.service = {k: v for k, v in message.items() if k != "type"}
            held = message.get("cameras") if isinstance(message.get("cameras"), dict) else {}
            # a stream the service has not been able to read for a while: said, and said again when it is back
            down = {cam for cam, info in held.items() if isinstance(info, dict) and info.get("stream") == "down" and cam in self.watching}
            was, self.stream_down = self.stream_down, down
            for camera in down - was:
                _LOGGER.info("A camera's stream is down: the service keeps trying, less and less often")
                self._say(camera, "offline", "stream_down")
            for camera in was - down:
                if self.roster.cameras.get(camera) == READY:
                    _LOGGER.info("A camera's stream is back")
                    self._say(camera, "online")
            for camera in down ^ was:
                self._seen_changed(camera)
            for camera, info in held.items():  # (the view, too, is said here now and then, should no packet have said it)
                if camera in self.watching and isinstance(info, dict) and isinstance(info.get("view"), dict):
                    self._view(camera, info["view"].get("state"))
            for camera in held:  # held by the service, but not by us (it went while we were not looking): let go
                if camera not in self.watching and camera not in self._waking:
                    self.link.send({"type": "sleep", "camera": camera})
        elif message.get("type") == "tracks" and isinstance(message.get("packet"), dict):
            camera, now = str(message.get("camera")), time.monotonic()
            if camera not in self.watching:
                return  # of a camera let go meanwhile: nothing of it is passed on or kept
            self._view(camera, message["packet"].get("view"))
            message["packet"].setdefault("view", self.view.get(camera, "steady"))  # (a service too old to say it)
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
            self._tell(camera, {**message["packet"], "mode": "auto"})

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
        return {**payload, "tracks": tracks, "source": "local" if served else "protect", "mode": "auto", "view": self.view.get(camera, "steady")}

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
    def subscribe(self, camera: str, send: Send, mode: str = "auto", on_end: Callable[[], None] | None = None) -> Callable[[], None]:
        """Have `send` called with a camera's packets. `on_end` is called if
        the subscription is ended from this side (the camera was removed)."""
        key = (camera, mode)
        self.listeners.setdefault(key, set()).add(send)
        if on_end is not None:
            self._ends[send] = on_end

        @callback
        def unsubscribe() -> None:
            self._ends.pop(send, None)
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
            "cameras": sorted(self.roster.cameras),
            "states": {cam: self.state(cam) for cam in sorted(self.roster.cameras)},
            "why": {cam: why for cam in sorted(self.roster.cameras) if (why := self.why(cam))},  # why each camera not watched is not
            "view": {cam: self.view.get(cam, "steady") for cam in sorted(self.watching)},
            "watching": {cam: {k: info[k] for k in ("entity", "channel", "since")} for cam, info in self.watching.items()},
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
    if camera not in hub.roster.cameras:
        connection.send_error(msg["id"], "not_ready", "This camera isn't one that is followed (it doesn't recognise things, or has just been removed)")
        return
    if mode == "auto" and hub._relay is None:  # noqa: SLF001
        connection.send_error(msg["id"], "not_ready", "Protect's live reports aren't being received (Protect Detections isn't loaded)")
        return
    if mode == "local" and not hub.link.connected:
        connection.send_error(msg["id"], "not_ready", "The tracking service isn't running")
        return

    @callback
    def send(payload: dict[str, Any]) -> None:
        connection.send_message(websocket_api.event_message(msg["id"], payload))

    connection.send_result(msg["id"])
    connection.subscriptions[msg["id"]] = hub.subscribe(camera, send, mode, lambda: connection.subscriptions.pop(msg["id"], None))
    send({"status": "connected", "mode": mode, "service": hub.link.connected, "view": hub.view.get(camera, "steady")})
    state = hub.state(camera)
    if state in (OFFLINE, NO_STREAM, "stream_down"):  # so that silence is not taken for an empty street
        send({"status": "offline", "mode": mode, "service": hub.link.connected, "view": "steady", "reason": state})


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
