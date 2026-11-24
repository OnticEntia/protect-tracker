"""Person seen: on while the tracking service itself sees a person on a camera.

It is the service's own word, not Protect's: it comes on as soon as the
service is sure of a person (about a quarter of a second of them walking),
which is as a rule before Protect reports one, and it comes on for people
Protect never reports. `confirmed` says whether Protect has reported the same
person meanwhile. It goes off a few seconds after the last sight of them.

It needs "Watch all the time" (the integration's options) to be of use:
otherwise the service only looks once Protect has already reported something.
One per camera that recognises things, on the camera's own device.

It is unavailable while the service is not reached, and while its camera is
offline or its stream cannot be read. A camera adopted later gets its sensor
without a restart. When a camera is removed from Protect, the UniFi Protect
integration removes the camera's device and this sensor goes with it; should
the device be left for any reason, the sensor removes itself all the same.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import TrackerHub
from .const import DOMAIN


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    hub: TrackerHub = hass.data[DOMAIN]
    made: set[str] = set()

    @callback
    def add(camera: str) -> None:
        mac = hub.macs.get(camera)
        if mac and camera not in made:
            made.add(camera)
            async_add_entities([PersonSeen(hub, camera, mac, entry.entry_id, lambda: made.discard(camera))])

    hub.add_sensor = add  # cameras adopted from here on
    entry.async_on_unload(lambda: setattr(hub, "add_sensor", None))
    for camera in list(hub.roster.cameras):
        add(camera)


class PersonSeen(BinarySensorEntity):
    _attr_has_entity_name = True
    _attr_name = "Person seen"
    _attr_device_class = BinarySensorDeviceClass.OCCUPANCY
    _attr_should_poll = False

    def __init__(self, hub: TrackerHub, camera: str, mac: str, entry_id: str, on_removed) -> None:
        self._hub = hub
        self._camera = camera
        self._entry_id = entry_id
        self._on_removed = on_removed
        self._attr_unique_id = f"{mac.replace(':', '').upper()}_person_seen"
        self._attr_device_info = DeviceInfo(connections={(dr.CONNECTION_NETWORK_MAC, dr.format_mac(mac))})

    @property
    def is_on(self) -> bool:
        return self._camera in self._hub.seen

    @property
    def available(self) -> bool:
        return self._hub.link.connected and self._hub.state(self._camera) in ("watching", "asleep")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        seen = self._hub.seen.get(self._camera) or {}
        return {"count": seen.get("count", 0), "confirmed": bool(seen.get("confirmed"))}

    async def async_added_to_hass(self) -> None:
        self._hub.seen_listeners.setdefault(self._camera, set()).add(self._changed)
        self._hub.gone_listeners.setdefault(self._camera, set()).add(self._gone)

    async def async_will_remove_from_hass(self) -> None:
        self._hub.seen_listeners.get(self._camera, set()).discard(self._changed)
        self._hub.gone_listeners.get(self._camera, set()).discard(self._gone)
        self._on_removed()

    @callback
    def _changed(self) -> None:
        self.async_write_ha_state()

    @callback
    def _gone(self) -> None:
        """The camera is no longer in Protect: the sensor is removed (if the
        camera's device has not taken it along already), and the device is
        not held on to on this integration's account."""
        entry = er.async_get(self.hass).async_get(self.entity_id) if self.entity_id else None
        if entry is None:
            return
        er.async_get(self.hass).async_remove(entry.entity_id)
        devices = dr.async_get(self.hass)
        if entry.device_id and devices.async_get(entry.device_id) is not None:
            devices.async_update_device(entry.device_id, remove_config_entry_id=self._entry_id)
