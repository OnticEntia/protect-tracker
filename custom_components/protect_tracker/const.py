"""Names shared by the integration's files."""

DOMAIN = "protect_tracker"
DETECTIONS = "protect_detections"  # the companion whose relay of Protect's live reports is listened to

CONF_URL = "url"
CONF_STREAM = "stream"
DEFAULT_URL = "ws://127.0.0.1:8791/link"
STREAMS = {"medium": 1, "high": 0}  # the camera's stream by name -> Protect's channel number
DEFAULT_STREAM = "medium"
CONF_ALWAYS = "always"  # watch every camera all the time, not only while Protect reports something on the move

# A report of Protect's is stamped this long after the frame it describes (ms, measured), by the camera's
# stream: -125 on the main stream (measured in the browser, against the picture), about -115 on the medium
# one (measured by the service, against its own boxes).
STAMP_MS = {0: -125.0, 1: -115.0, 2: -115.0}
