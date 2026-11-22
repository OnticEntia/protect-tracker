"""The connection to the tracking service.

Home Assistant is the one that connects (a websocket), so the service needs
no login to Home Assistant and none to Protect: it is told what to watch and
is handed Protect's reports, and it answers with where things are. While the
service can't be reached this tries again now and then, and what was to be
sent meanwhile is dropped (it is live data: old reports are of no use).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import logging
from typing import Any

from aiohttp import ClientError, WSMsgType

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_LOGGER = logging.getLogger(__name__)

WAITING = 500  # messages kept while the socket is busy; beyond that the oldest go


class ServiceLink:
    def __init__(self, hass: HomeAssistant, url: str, on_message: Callable[[dict[str, Any]], None], on_connect: Callable[[], None]) -> None:
        self.hass = hass
        self.url = url
        self.on_message = on_message
        self.on_connect = on_connect
        self.connected = False
        self.error: str | None = None
        self._task: asyncio.Task | None = None
        self._out: asyncio.Queue[str] = asyncio.Queue()

    def start(self) -> None:
        self._task = self.hass.async_create_background_task(self._run(), "protect_tracker link")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self.connected = False

    def send(self, message: dict[str, Any]) -> bool:
        """Queue a message for the service. False when it isn't connected."""
        if not self.connected:
            return False
        while self._out.qsize() >= WAITING:
            self._out.get_nowait()
        self._out.put_nowait(json.dumps(message))
        return True

    async def _run(self) -> None:
        backoff = 2.0
        while True:
            try:
                await self._connection()
                backoff = 2.0
            except asyncio.CancelledError:
                raise
            except (ClientError, OSError, asyncio.TimeoutError) as err:
                self.error = type(err).__name__
            except Exception:  # noqa: BLE001
                _LOGGER.debug("The link to the tracking service failed", exc_info=True)
                self.error = "error"
            self.connected = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def _connection(self) -> None:
        session = async_get_clientsession(self.hass)
        async with session.ws_connect(self.url, heartbeat=30, max_msg_size=0) as ws:
            while not self._out.empty():
                self._out.get_nowait()
            self.connected = True
            self.error = None
            self.on_connect()
            sender = asyncio.create_task(self._sender(ws))
            try:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    try:
                        data = json.loads(msg.data)
                    except ValueError:
                        continue
                    if isinstance(data, dict):
                        try:
                            self.on_message(data)
                        except Exception:  # noqa: BLE001 - one bad message is no reason to drop the link
                            _LOGGER.debug("A message from the tracking service couldn't be used", exc_info=True)
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)

    async def _sender(self, ws) -> None:
        while True:
            await ws.send_str(await self._out.get())
