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
without a restart, as soon as the UniFi Protect integration has made its
device. When a camera is removed from Protect the sensor removes itself, and
with it this integration's hold on the camera's device (where the device is
shared with the UniFi Protect integration, that one's removing it takes the
sensor along first).
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import async_generate_entity_id
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import TrackerHub
from .const import DOMAIN


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    hub: TrackerHub = hass.data[DOMAIN]
    made: set[str] = set()

    @callback
    def add(camera: str) -> None:
        if camera in made:
            return
        mac = hub.macs.get(camera)
        # (the sensor belongs on the camera's own device, which the UniFi Protect integration makes: a camera
        # just adopted may not have it yet, and a device made here would have no name to call the sensor by)
        device = _their_device(hass, mac) if mac else None
        if device is not None:
            made.add(camera)
            sensor = PersonSeen(hub, camera, mac, entry.entry_id, lambda: made.discard(camera), device)
            name = device.name_by_user or device.name
            if name:  # its id says which camera it is of (one already there keeps the id it has)
                sensor.entity_id = async_generate_entity_id("binary_sensor.{}", f"{name} person seen", hass=hass)
            async_add_entities([sensor])

    hub.add_sensor = add  # cameras adopted from here on
    entry.async_on_unload(lambda: setattr(hub, "add_sensor", None))
    for camera in list(hub.roster.cameras):
        add(camera)


def _their_device(hass: HomeAssistant, mac: str) -> dr.DeviceEntry | None:
    """The camera's device as the UniFi Protect integration has made it."""
    devices = dr.async_get(hass)
    for theirs in hass.config_entries.async_entries("unifiprotect"):
        device = devices.async_get_device_by_connection((dr.CONNECTION_NETWORK_MAC, dr.format_mac(mac)), theirs.entry_id)
        if device is not None:
            return device
    return None


class PersonSeen(BinarySensorEntity):
    _attr_has_entity_name = True
    _attr_name = "Person seen"
    _attr_device_class = BinarySensorDeviceClass.OCCUPANCY
    _attr_should_poll = False

    def __init__(self, hub: TrackerHub, camera: str, mac: str, entry_id: str, on_removed, theirs: dr.DeviceEntry | None = None) -> None:
        self._hub = hub
        self._camera = camera
        self._entry_id = entry_id
        self._on_removed = on_removed
        self._attr_unique_id = f"{mac.replace(':', '').upper()}_person_seen"
        # The camera, by its hardware address. Where Home Assistant keeps a device per integration, this is a
        # device of this integration's own beside the UniFi Protect one, so it is given the camera's name.
        self._attr_device_info = DeviceInfo(connections={(dr.CONNECTION_NETWORK_MAC, dr.format_mac(mac))})
        if theirs is not None and theirs.name:
            self._attr_device_info.update(name=theirs.name, manufacturer=theirs.manufacturer, model=theirs.model)

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
        camera's device has not taken it along already), and so is this
        integration's hold on the device (its own device, where it has one)."""
        entry = er.async_get(self.hass).async_get(self.entity_id) if self.entity_id else None
        if entry is None:
            return
        er.async_get(self.hass).async_remove(entry.entity_id)
        devices = dr.async_get(self.hass)
        device = devices.async_get(entry.device_id) if entry.device_id else None
        if device is None:
            return
        if device.primary_config_entry == self._entry_id:  # this integration's own device of the camera
            devices.async_remove_device(device.id)
        else:  # (a Home Assistant where the device is shared with the UniFi Protect integration: only our part in it)
            devices.async_update_device(device.id, remove_config_entry_id=self._entry_id)
