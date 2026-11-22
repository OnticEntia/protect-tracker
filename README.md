# Protect Tracker

Follows what UniFi Protect reports, frame by frame, with a model running on your own machine.

UniFi Protect stays the authority on what is there: its detections, types, zones and events are
untouched. Protect's live boxes arrive about four times a second; this project finds the same
things in the camera's video on every frame, so the boxes drawn over a live picture sit on what
they mark. UniFi Protect cameras only.

**Status: early.** One camera at a time has been followed live; the boxes reach Home Assistant through
`protect_tracker/live/subscribe`. No card of this project's own yet: Protect Detections' tracking card
and camera wall are the ones meant to draw them.

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

## Installing

1. **The service.** It runs next to Home Assistant, as a container: see `service/compose.example.yaml`.
   Put the model files in a `models` folder and mount it at `/models`. It listens on this machine only
   (`127.0.0.1:8791`), has no login of its own, and can do nothing but look at the streams Home Assistant
   hands it. With an NVIDIA card (and the NVIDIA container toolkit) it uses the card; without, the CPU.
2. **The integration.** Install this repository through HACS as a custom repository (type: integration),
   restart Home Assistant, and add "Protect Tracker" under Settings → Devices & services. It needs the
   UniFi Protect integration and [Protect Detections](https://github.com/OnticEntia/protect-detections).

The service is woken when a camera reports something on the move and left alone again half a minute
after the last movement; things that only stand there (a parked car) do not wake it. With "Watch all
the time" (the integration's options) every camera is watched always instead: while nothing moves the
service looks at a few frames a second, and at every frame as soon as something does.

## For cards: `protect_tracker/live/subscribe`

    {"type": "protect_tracker/live/subscribe", "entity": "<a UniFi Protect camera entity>"}

Messages have the shape of Protect Detections' `protect_detections/live/subscribe` (`clockWall`,
`clockStream`, `clockStreamRate`, `eventId`, `tracks[]` with `id`, `coord` [x, y, w, h] in 0–1000,
`objectType`, `confidence`, `stationary`, `zones`, `loiterZones`, `lines`, `desc`), and `{"status": ...}`
messages are passed on as they are. What is added:

- **`source`** on every packet with boxes: `"local"` for boxes found by the service (each track also has
  `local: true`), `"protect"` for Protect's own, passed on as Protect sent them.
- The service's boxes are **stamped as Protect stamps its own** (a little after the frame they belong
  to), so a card times both the same way and needs no change for it.
- A thing the service follows gets a packet for every frame, under Protect's track id and with Protect's
  details. Protect's reports of it keep coming too; they are older than what the card already has, so
  their boxes fall away by themselves. When the service loses the thing, or is not running, Protect's
  boxes are what is left: nothing has to be switched.
- **Faces.** Where the service finds the face of a person it is following and Protect reports no face,
  the face is a track of its own: `id` is `<the person's track id>:face`, `objectType` is `face`, and
  `of` names the person's track. It is the one thing sent that Protect did not report itself.

The command answers `not_loaded`, `not_found` or `not_ready` when it cannot serve a camera; a card
should then use Protect Detections' command as before.

## Development

    python -m venv .venv && . .venv/bin/activate
    pip install "onnxruntime-gpu[cuda,cudnn]" numpy av aiohttp pytest
    pytest tests                       # the service: plain Python
    pytest tests_ha -o asyncio_mode=auto   # the integration: where Home Assistant and
                                           # pytest-homeassistant-custom-component are installed

`service/tools/spike.py` measures how the service's own boxes line up with Protect's.

## Licence

MIT.
