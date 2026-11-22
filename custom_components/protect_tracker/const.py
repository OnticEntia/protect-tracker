"""Names shared by the integration's files."""

DOMAIN = "protect_tracker"
DETECTIONS = "protect_detections"  # the companion whose relay of Protect's live reports is listened to

CONF_URL = "url"
CONF_STREAM = "stream"
DEFAULT_URL = "ws://127.0.0.1:8787/link"
STREAMS = {"medium": 1, "high": 0}  # the camera's stream by name -> Protect's channel number
DEFAULT_STREAM = "medium"
