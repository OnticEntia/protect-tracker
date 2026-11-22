"""python -m tracker_service: the tracking service."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from aiohttp import web

from .camera import Engine
from .server import Service


def main() -> None:
    ap = argparse.ArgumentParser(description="Follows what UniFi Protect reports, frame by frame.")
    ap.add_argument("--host", default=os.environ.get("TRACKER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("TRACKER_PORT", "8791")))
    ap.add_argument("--models", default=os.environ.get("TRACKER_MODELS", "models"), help="the folder the model files are in")
    ap.add_argument("--model", default=os.environ.get("TRACKER_MODEL", "m"), help="YOLOX size: nano, tiny, s, m, l, x")
    ap.add_argument("--no-faces", action="store_true", default=os.environ.get("TRACKER_FACES", "1") == "0")
    ap.add_argument("--no-far", action="store_true", default=os.environ.get("TRACKER_FAR", "1") == "0", help="without the sharper look for people far off")
    args = ap.parse_args()
    logging.basicConfig(level=os.environ.get("TRACKER_LOG", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    models = Path(args.models)
    if not (models / f"yolox_{args.model}.onnx").exists():
        raise SystemExit(f"No yolox_{args.model}.onnx in the models folder (see the README for where to get it)")
    engine = Engine(models, args.model, faces=not args.no_faces, far=not args.no_far)
    logging.info("Detector yolox_%s on the %s%s; faces %s; far look %s", args.model, engine.device, " (fast copy)" if engine.detector.fast else "",
                 "on" if engine.face else "off", "on" if engine.far else "off")
    web.run_app(Service(engine).app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
