"""A stream larger than anything looks at (a camera's main stream) is brought
down as it is decoded: on the graphics card where it has a video decoder,
else in software. Either way the frames keep their timestamps."""

from fractions import Fraction

import numpy as np
import pytest

from fakes import packetize
from tracker_service.rtp import Depacketizer, parse_rtp
from tracker_service.video import MOST, Decoder, fits, picture

W, H, STEP = 1920, 1080, 30  # a made-up main stream; its square moves 30 points a frame


def coded(codec: str, count: int, width: int = W, height: int = H) -> list[bytes]:
    import av

    ctx = av.CodecContext.create("libx265" if codec == "h265" else "libx264", "w")
    ctx.width, ctx.height, ctx.pix_fmt = width, height, "yuv420p"
    ctx.time_base = Fraction(1, 30)
    ctx.options = (
        {"preset": "ultrafast", "tune": "zerolatency", "x265-params": "bframes=0:log-level=none:repeat-headers=1:keyint=10"}
        if codec == "h265"
        else {"preset": "ultrafast", "tune": "zerolatency", "g": "10"}
    )
    out = []
    for i in range(count):
        img = np.zeros((height, width, 3), np.uint8)
        img[height // 2 - 60 : height // 2 + 60, STEP * i : STEP * i + 120] = 255
        frame = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
        frame.pts = i
        out += [bytes(p) for p in ctx.encode(frame)]
    return out + [bytes(p) for p in ctx.encode(None)]


def which(frame, width: int = W) -> int:
    """Which frame a decoded picture is, from where its square is (at whatever size it came out)."""
    img = picture(frame)
    cols = np.where(img[img.shape[0] // 2, :, 1] > 128)[0]
    return round(cols[0] * width / img.shape[1] / STEP) if len(cols) else -1


def decode_all(codec: str, decoder: Decoder, count: int = 24, width: int = W, height: int = H):
    """[(timestamp, frame, when its packet came as the decoder has it)]; packets "arrive" a second apart."""
    depack, got = Depacketizer(codec), []
    for raw in packetize(codec, coded(codec, count, width, height), 0, mtu=1200):
        for n, unit in enumerate(depack.push(parse_rtp(raw))):
            at = unit.timestamp / 3000 * 1000.0
            for stamp, frame in decoder.decode(unit, at):
                got.append((stamp, frame, decoder.came(stamp, at)))
    return got


def test_the_size_a_picture_is_brought_down_to():
    assert fits(1280, 720) is None and fits(640, 360) is None
    assert fits(3840, 2160) == (1280, 720) and fits(1920, 1080) == (1280, 720)
    assert fits(2688, 1512) == (1280, 720)
    assert fits(2560, 1920) == (960, 720)  # a taller picture keeps its shape


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_a_large_stream_is_brought_down_in_software(codec):
    decoder = Decoder(codec, gpu=False)
    got = decode_all(codec, decoder)
    assert decoder.how == "software" and decoder.size == (W, H)
    assert len(got) >= 22
    for stamp, frame, at in got:
        assert (frame.width, frame.height) == MOST
        assert stamp == which(frame) * 3000, "a frame came out stamped as another"
        assert at == stamp / 3000 * 1000.0


def test_a_stream_small_enough_is_left_as_it_is():
    decoder = Decoder("h264")
    got = decode_all("h264", decoder, width=1280, height=720)
    assert decoder.how == "software" and decoder.size == (1280, 720) and decoder.gpu_error is None
    assert all((frame.width, frame.height) == (1280, 720) for _, frame, _ in got)
    assert all(stamp == which(frame, 1280) * 3000 for stamp, frame, _ in got)


def test_without_the_cards_decoder_software_does_it(monkeypatch):
    def none(self, size):
        raise RuntimeError("no such decoder")

    monkeypatch.setattr(Decoder, "_on_card", none)
    decoder = Decoder("h265")
    got = decode_all("h265", decoder)
    assert decoder.how == "software" and decoder.gpu_error == "RuntimeError"
    assert len(got) >= 22 and all((frame.width, frame.height) == MOST and stamp == which(frame) * 3000 for stamp, frame, _ in got)


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_a_large_stream_is_brought_down_on_the_card_where_it_can_be(codec):
    decoder = Decoder(codec)
    got = decode_all(codec, decoder)
    if decoder.how != "gpu":
        pytest.skip(f"no video decoder on a graphics card here ({decoder.gpu_error})")
    assert decoder.size == (W, H) and len(got) >= 21
    for stamp, frame, at in got:
        assert (frame.width, frame.height) == MOST
        assert stamp == which(frame) * 3000, "a frame came out stamped as another"
        assert at == stamp / 3000 * 1000.0  # the time its own packet came, though it came out a packet later
    assert [stamp for stamp, _, _ in got] == sorted(stamp for stamp, _, _ in got)


def test_a_kept_scaler_gives_the_same_pictures_as_one_made_for_each_frame():
    from tracker_service.video import small_grey

    got = decode_all("h264", Decoder("h264", most=None), count=6)
    for _, frame, _ in got:  # (frames of the stream's own size)
        assert (frame.width, frame.height) == (W, H)
        assert np.array_equal(picture(frame, 640, 360), frame.to_ndarray(format="bgr24", width=640, height=360))
        assert np.array_equal(picture(frame), frame.to_ndarray(format="bgr24"))
        grey = small_grey(frame, 160, 90)
        assert grey.shape == (90, 160) and grey.dtype == np.uint8
        careful = frame.reformat(width=160, height=90, format="gray", interpolation="AREA").to_ndarray()
        assert np.array_equal(grey, careful)


def test_scalers_are_kept_apart_by_thread_and_by_what_they_make():
    import threading

    from tracker_service import video

    frame = decode_all("h264", Decoder("h264", most=None), count=2)[0][1]
    video.picture(frame, 640, 360), video.picture(frame, 320, 180), video.picture(frame, 640, 360)
    mine = dict(video._scalers.kept)
    assert {(640, 360, "bgr24", None), (320, 180, "bgr24", None)} <= set(mine)
    theirs = {}
    worker = threading.Thread(target=lambda: (video.picture(frame, 640, 360), theirs.update(video._scalers.kept)))
    worker.start()
    worker.join()
    assert theirs[(640, 360, "bgr24", None)] is not mine[(640, 360, "bgr24", None)]


def test_told_to_watch_another_stream_the_first_is_closed_and_each_is_decoded_as_fits_it():
    """A camera first watched by its main stream (the only one there) and then
    by its medium one: the service says of each how it decodes it, and hangs
    up on the first when it is given the second."""
    import asyncio

    import aiohttp
    from aiohttp import web

    from fakes import FakeCamera, encode
    from test_service import FIRST, SquareFinder
    from tracker_service.server import Service

    async def until(what, timeout: float = 5.0):
        end = asyncio.get_running_loop().time() + timeout
        while not what():
            assert asyncio.get_running_loop().time() < end, "it never came"
            await asyncio.sleep(0.02)

    async def go():
        service = Service(SquareFinder())
        runner = web.AppRunner(service.app())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        main = FakeCamera("h265", packetize("h265", coded("h265", 12), FIRST, mtu=1200), pace=1 / 30, path="/MAIN")
        medium = FakeCamera("h265", packetize("h265", encode("h265", 12), FIRST, mtu=600), pace=1 / 30, path="/MEDIUM")
        main.stay = medium.stay = True
        async with main, medium, aiohttp.ClientSession() as session:
            ws = await session.ws_connect(f"ws://127.0.0.1:{port}/link")
            await ws.send_json({"type": "watch", "camera": "c1", "source": main.url, "channel": 0, "stamp": -125})
            await until(lambda: (service.status()["cameras"].get("c1") or {}).get("frames", 0) >= 8)
            first = service.status()["cameras"]["c1"]
            assert first["decode"]["size"] == [W, H] and first["decode"]["how"] in ("gpu", "software")
            assert first["decode"]["ms"] is not None and main.gone == 0
            worker = service.cameras["c1"].worker
            await ws.send_json({"type": "watch", "camera": "c1", "source": medium.url, "channel": 1, "stamp": -115})
            await until(lambda: main.gone == 1)  # hung up on the main stream
            await until(lambda: (service.status()["cameras"].get("c1") or {}).get("frames", 0) >= 8 and service.cameras["c1"].worker is not worker)
            second = service.status()["cameras"]["c1"]
            assert second["decode"] == {"how": "software", "size": [320, 240], "ms": second["decode"]["ms"], "gpu_error": None}
            assert not worker.is_alive() and medium.gone == 0 and len(service.cameras) == 1
            await ws.send_json({"type": "sleep", "camera": "c1"})
            await until(lambda: medium.gone == 1)
            await ws.close()
        await runner.cleanup()

    asyncio.run(go())
