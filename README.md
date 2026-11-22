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

- **`local`** on every track: `true` for a box found by the service, `false` for Protect's own.
- **One box a track.** A thing the service follows gets a packet for every frame, under Protect's track
  id and with Protect's details; while those are fresh, Protect's own box for that id is left out of what
  is passed on. When the service loses the thing, Protect's box comes back after a moment. Things the
  service does not follow (parked cars, kinds it has no model for) are Protect's throughout.
- **`stationary`** is Protect's word, passed on as Protect said it, on every track.
- **`moving`** is the service's own view, and is only on its own tracks (`local: true`; a track of
  Protect's passed on has no such field). It is `true` when the middle of the thing's box, as the
  service found it, is at least a hundredth of the picture (width and height counted together) from
  where it was a second before, or at the start of the track if that is less than a second ago; `false`
  otherwise, and `false` on a track's very first box. It is worked out afresh for every frame.
- **`source`** on a packet says how the camera is being served just then and stays put: `"local"` while
  the service is watching it, `"protect"` while it is not (asleep, or not running).
- The service's boxes are **stamped as Protect stamps its own** (a little after the frame they belong
  to), so a card times both the same way and needs no change for it.
- **Faces.** Where the service finds the face of a person it is following and Protect reports no face,
  the face is a track of its own: `id` is `<the person's track id>:face`, `objectType` is `face`, and
  `of` names the person's track. It is the one thing sent that Protect did not report itself.

### `mode: "local"`: on the service's own word

    {"type": "protect_tracker/live/subscribe", "entity": "<camera entity>", "mode": "local"}

Without `mode` (or with `"auto"`) the command is as described above: Protect decides what is there and
the service refines where. With `"local"` the service is the authority. It tells what it sees as soon
as it is sure, whether or not Protect has reported anything, and goes on doing so when Protect is
silent. Nothing of Protect's own is passed on in this mode. Same packets; per track:

| Field | |
|---|---|
| `id` | the service's own track id (starts with `L`), the same for the life of the track, not used again after a restart |
| `confirmed` | `false` while only the service has seen the thing; `true` from the moment Protect reports the same thing, and for the rest of the track |
| `protectId` | Protect's track id, once confirmed |
| `objectType`, `confidence` | the service's (person, vehicle, animal, face; 0–100) until confirmed, Protect's after |
| `desc`, `zones`, `loiterZones`, `lines` | empty until confirmed, then Protect's, kept as last reported |
| `stationary` | `false` until confirmed, then Protect's word |
| `moving`, `local` | as above (`local` is always `true`) |

A face is a track of its own here too (`<the person's id>:face`, `of`).

With nobody to veto it, the service only tells a thing that has been found for a quarter of a second,
surely enough and big enough for its kind, **and has travelled from where it was first found**. What
only stands there (a parked car, a bike) is never told; a parked car that drives off is told from when
it moves. Packets are stamped as in the default mode, so a card times them the same way. Before
Protect has reported anything since the service started, `clockWall` is the service's own estimate
(`clockStream`, which is what the picture is matched with, is exact from the first packet).

This mode wants "Watch all the time" switched on: otherwise the service only looks once Protect has
already reported something.

The command answers `not_loaded`, `not_found` or `not_ready` when it cannot serve a camera; a card
should then use Protect Detections' command as before.

## Entities

| Entity | |
|---|---|
| **Person seen** (binary sensor, one per camera, on the camera's device) | On while the service itself sees a person on that camera: as a rule before Protect reports one, and also for people Protect never reports. Attributes: `count`, and `confirmed` (Protect has reported the same person meanwhile). Off a few seconds after the last sight. Needs "Watch all the time". |

Use it as a trigger in your own automations where being early matters more than being certain;
Protect's own detections remain the certain ones.

## People far off

Five times a second the service also takes a sharper look for people: the picture at twice the
detector's size, cut into four tiles, each looked at on its own, so someone far away is four times
the pixels. It costs about as much as five ordinary looks each time. `TRACKER_FAR=0` turns it off.

## Development

    python -m venv .venv && . .venv/bin/activate
    pip install "onnxruntime-gpu[cuda,cudnn]" numpy av aiohttp pytest
    pytest tests                       # the service: plain Python
    pytest tests_ha -o asyncio_mode=auto   # the integration: where Home Assistant and
                                           # pytest-homeassistant-custom-component are installed

`service/tools/spike.py` measures how the service's own boxes line up with Protect's.

## Licence

MIT.
