"""Setup: where the tracking service is, and which of a camera's streams it follows."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers.selector import SelectSelector, SelectSelectorConfig

from .const import CONF_ALWAYS, CONF_STREAM, CONF_URL, DEFAULT_STREAM, DEFAULT_URL, DOMAIN, STREAMS


def _schema(url: str, stream: str, always: bool = False) -> vol.Schema:
    return vol.Schema({
        vol.Required(CONF_URL, default=url): str,
        vol.Required(CONF_STREAM, default=stream): SelectSelector(
            SelectSelectorConfig(options=list(STREAMS), translation_key="stream")
        ),
        vol.Required(CONF_ALWAYS, default=always): bool,
    })


class ProtectTrackerFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return ProtectTrackerOptions()

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=_schema(DEFAULT_URL, DEFAULT_STREAM))
        return self.async_create_entry(title="Protect Tracker", data={}, options=user_input)


class ProtectTrackerOptions(OptionsFlow):
    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        now = self.config_entry.options
        return self.async_show_form(
            step_id="init",
            data_schema=_schema(now.get(CONF_URL, DEFAULT_URL), now.get(CONF_STREAM, DEFAULT_STREAM), bool(now.get(CONF_ALWAYS, False))),
        )
