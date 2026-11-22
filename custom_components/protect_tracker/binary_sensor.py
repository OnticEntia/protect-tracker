"""Person seen: on while the tracking service itself sees a person on a camera.

It is the service's own word, not Protect's: it comes on as soon as the
service is sure of a person (about a quarter of a second of them walking),
which is as a rule before Protect reports one, and it comes on for people
Protect never reports. `confirmed` says whether Protect has reported the same
person meanwhile. It goes off a few seconds after the last sight of them.

It needs "Watch all the time" (the integration's options) to be of use:
otherwise the service only looks once Protect has already reported something.
One per camera that recognises things, on the camera's own device.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import TrackerHub, _protect_api, smart_cameras
from .const import DOMAIN


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    hub: TrackerHub = hass.data[DOMAIN]
    api = _protect_api(hass)
    cameras = api.bootstrap.cameras if api is not None else {}
    async_add_entities(PersonSeen(hub, cameras[cam]) for cam in smart_cameras(api) if cameras[cam].mac)


class PersonSeen(BinarySensorEntity):
    _attr_has_entity_name = True
    _attr_name = "Person seen"
    _attr_device_class = BinarySensorDeviceClass.OCCUPANCY
    _attr_should_poll = False

    def __init__(self, hub: TrackerHub, camera: Any) -> None:
        self._hub = hub
        self._camera = camera.id
        mac = camera.mac.replace(":", "").upper()
        self._attr_unique_id = f"{mac}_person_seen"
        self._attr_device_info = DeviceInfo(connections={(dr.CONNECTION_NETWORK_MAC, dr.format_mac(camera.mac))})

    @property
    def is_on(self) -> bool:
        return self._camera in self._hub.seen

    @property
    def available(self) -> bool:
        return self._hub.link.connected

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        seen = self._hub.seen.get(self._camera) or {}
        return {"count": seen.get("count", 0), "confirmed": bool(seen.get("confirmed"))}

    async def async_added_to_hass(self) -> None:
        self._hub.seen_listeners.setdefault(self._camera, set()).add(self._changed)

    async def async_will_remove_from_hass(self) -> None:
        self._hub.seen_listeners.get(self._camera, set()).discard(self._changed)

    @callback
    def _changed(self) -> None:
        self.async_write_ha_state()
