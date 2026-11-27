"""python -m tracker_service: the tracking service."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from aiohttp import web

from .camera import Engine
from .pace import MIN_FPS
from .server import Service


def main() -> None:
    ap = argparse.ArgumentParser(description="Follows people, vehicles and animals on UniFi Protect cameras, frame by frame.")
    ap.add_argument("--host", default=os.environ.get("TRACKER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("TRACKER_PORT", "8791")))
    ap.add_argument("--models", default=os.environ.get("TRACKER_MODELS", "models"), help="the folder the model files are in")
    ap.add_argument("--model", default=os.environ.get("TRACKER_MODEL", "m"), help="YOLOX size: nano, tiny, s, m, l, x")
    ap.add_argument("--no-faces", action="store_true", default=os.environ.get("TRACKER_FACES", "1") == "0")
    ap.add_argument("--no-far", action="store_true", default=os.environ.get("TRACKER_FAR", "1") == "0", help="without the sharp look (the picture at twice the size)")
    ap.add_argument("--idle-every", type=int, default=int(os.environ.get("TRACKER_IDLE_EVERY", "1")),
                    help="while nothing moves, look at one frame in this many (1: every frame; 6: five a second)")
    ap.add_argument("--far-every", type=int, default=int(os.environ.get("TRACKER_FAR_EVERY", "1")),
                    help="the sharp look on one frame in this many (1: every frame, while the floor allows)")
    ap.add_argument("--min-fps", type=float, default=float(os.environ.get("TRACKER_MIN_FPS", str(MIN_FPS))),
                    help=f"the floor: frames of each camera looked at a second, whatever else gives way ({MIN_FPS} at least)")
    ap.add_argument("--no-gpu-decode", action="store_true", default=os.environ.get("TRACKER_GPU_DECODE", "1") == "0",
                    help="decode in software always (else a stream larger than 1280x720 is decoded on the graphics card where it can be)")
    args = ap.parse_args()
    if args.min_fps < MIN_FPS:
        ap.error(f"the floor (--min-fps / TRACKER_MIN_FPS) cannot be under {MIN_FPS} frames a second")
    logging.basicConfig(level=os.environ.get("TRACKER_LOG", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    models = Path(args.models)
    if not (models / f"yolox_{args.model}.onnx").exists():
        raise SystemExit(f"No yolox_{args.model}.onnx in the models folder (see the README for where to get it)")
    engine = Engine(models, args.model, faces=not args.no_faces, far=not args.no_far, idle_every=args.idle_every, far_every=args.far_every, min_fps=args.min_fps,
                    gpu_decode=not args.no_gpu_decode)
    logging.info("Detector yolox_%s on the %s%s; faces %s; sharp look %s", args.model, engine.device, " (fast copy)" if engine.detector.fast else "",
                 "on" if engine.face else "off", "on" if engine.far else "off")
    web.run_app(Service(engine).app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
