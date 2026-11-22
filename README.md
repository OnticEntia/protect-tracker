# Protect Tracker

Follows what UniFi Protect reports, frame by frame, with a model running on your own machine.

UniFi Protect stays the authority on what is there: its detections, types, zones and events are
untouched. Protect's live boxes arrive about four times a second; this project finds the same
things in the camera's video on every frame, so the boxes drawn over a live picture sit on what
they mark. UniFi Protect cameras only.

**Status: early.** The video path, the detectors and the link to Home Assistant exist and are
tested; local boxes are not yet sent to any card.

## The parts

| Part | Where | What it does |
|---|---|---|
| Integration | `custom_components/protect_tracker` | The go-between inside Home Assistant. Listens to Protect's live reports (through [Protect Detections](https://github.com/OnticEntia/protect-detections)), wakes the service when a camera reports something on the move, hands it the camera's stream and the reports. Holds no model. |
| Service | `service/` | Reads the camera's video (keeping each frame's own timestamp, which is what ties a box to its frame), runs the detectors, follows things. Runs next to Home Assistant, on a graphics card when there is one. |

Nothing logs in to Protect on its own and nothing is written to it: everything comes through the
UniFi Protect integration you already have. The camera's stream address is passed to the service
in memory when it is needed and is never stored or logged.

## No footage is stored

This project never writes video or pictures to disk: no recordings, no clips, no buffer files,
no saved frames. A frame lives in memory for as long as it takes to look at it. Your console
already records everything; keeping a second copy elsewhere would only be one more place for
footage of your home to sit. Anything that needs the past reads it from the console's own
recordings, through Home Assistant, tied to Protect's event.

What is kept is derived data only: boxes, tracks, timings, ids and measurements.

## Models

The model files are not part of this repository. The service reads them from a models folder.

| Model | For | Licence | Source |
|---|---|---|---|
| YOLOX (nano … x) | people, vehicles, animals | Apache-2.0 | the YOLOX project's releases (ONNX) |
| YuNet | faces | MIT | OpenCV's model zoo |

Run through ONNX Runtime (MIT). On a graphics card a half-precision copy of the YOLOX model with
an input only as tall as a 16:9 frame is preferred where there is one (`<models>/fast/`).

## Development

    python -m venv .venv && . .venv/bin/activate
    pip install "onnxruntime-gpu[cuda,cudnn]" numpy av aiohttp pytest
    pytest tests                       # the service: plain Python
    pytest tests_ha -o asyncio_mode=auto   # the integration: where Home Assistant and
                                           # pytest-homeassistant-custom-component are installed

`service/tools/spike.py` measures how the service's own boxes line up with Protect's.

## Licence

MIT.
