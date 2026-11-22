"""Tests that need Home Assistant itself. Run them where it is installed
(with pytest-homeassistant-custom-component):

    pytest tests_ha -o asyncio_mode=auto
"""

import pytest


@pytest.fixture(autouse=True)
def _custom(enable_custom_integrations):
    yield
