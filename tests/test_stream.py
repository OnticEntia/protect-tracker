"""The video path: RTSP -> RTP -> pictures, each with the timestamp it was sent with."""

import asyncio

import pytest

from fakes import FakeCamera, encode, packetize, rtp, square_at
from tracker_service.rtp import Depacketizer, Unwrap, parse_rtp
from tracker_service.rtsp import RtspError, RtspReader, parse_sdp
from tracker_service.video import Decoder, picture

NEAR_WRAP = (1 << 32) - 3000 * 5  # the timestamps start over during the test


async def read_all(url: str):
    frames = []
    async with RtspReader(url) as reader:
        depack = Depacketizer(reader.track.codec)
        decoder = Decoder(reader.track.codec, reader.track.parameter_sets)
        async for packet in reader.packets():
            for unit in depack.push(packet):
                assert unit.whole
                frames += decoder.decode(unit)
    return reader.track, frames


@pytest.mark.parametrize("codec", ["h264", "h265"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_frames_keep_their_timestamps(codec, aggregate):
    async def go():
        coded = encode(codec, 20)
        async with FakeCamera(codec, packetize(codec, coded, NEAR_WRAP, aggregate=aggregate)) as cam:
            return await read_all(cam.url)

    track, frames = asyncio.run(go())
    assert track.codec == codec
    assert len(frames) >= 18
    for ts, img in frames:
        i = square_at(picture(img))
        assert ts == (NEAR_WRAP + i * 3000) % (1 << 32), f"frame {i} came out stamped as another"


def test_starts_at_a_key_picture():
    coded = encode("h264", 20)
    packets = packetize("h264", coded, 0)
    depack, decoder, got = Depacketizer("h264"), Decoder("h264"), []
    started = False
    for raw in packets:
        packet = parse_rtp(raw)
        if not started and packet.timestamp < 3 * 3000:
            continue  # joined in the middle
        started = True
        for unit in depack.push(packet):
            got += decoder.decode(unit)
    assert got and square_at(picture(got[0][1])) == 10  # the next key picture (every tenth)


def test_a_lost_packet_is_noticed():
    coded = encode("h264", 12)
    packets = [parse_rtp(p) for p in packetize("h264", coded, 0, mtu=100)]
    victim = next(i for i, p in enumerate(packets) if p.timestamp == 3000 * 4 and not p.marker)
    depack, units = Depacketizer("h264"), []
    for i, packet in enumerate(packets):
        if i != victim:
            units += depack.push(packet)
    broken = [u.timestamp // 3000 for u in units if not u.whole]
    assert broken == [4]


def test_rtp_header_extras():
    plain = parse_rtp(rtp(b"\x65abc", 7, 1234, True))
    assert (plain.sequence, plain.timestamp, plain.marker, plain.payload) == (7, 1234, True, b"\x65abc")
    raw = bytearray(rtp(b"\x65abc", 7, 1234, False))
    raw[0] |= 0x10 | 0x20  # an extension, and padding
    raw = bytes(raw[:12]) + b"\xbe\xde\x00\x01ABCD" + b"\x65abc" + b"\x00\x00\x03"
    assert parse_rtp(raw).payload == b"\x65abc"
    assert parse_rtp(b"\x00" * 12) is None and parse_rtp(b"short") is None


def test_unwrap_counts_on():
    u = Unwrap()
    top = (1 << 32) - 3000
    assert [u(top), u(0), u(3000), u(0)] == [top, top + 3000, top + 6000, top + 3000]


def test_sdp():
    track = parse_sdp(
        "v=0\r\nm=audio 0 RTP/AVP 97\r\na=control:a\r\nm=video 0 RTP/AVP 98\r\na=rtpmap:98 H265/90000\r\n"
        "a=fmtp:98 sprop-vps=QAEMAf//;sprop-sps=QgEB;sprop-pps=RAHA\r\na=control:trackID=3\r\n"
    )
    assert (track.codec, track.payload_type, track.control, len(track.parameter_sets)) == ("h265", 98, "trackID=3", 3)
    assert parse_sdp("v=0\r\nm=audio 0 RTP/AVP 97\r\n") is None


def test_errors_do_not_name_the_address():
    async def go():
        async with FakeCamera("h264", []) as cam:
            secret = f"rtsp://user:hunter2@127.0.0.1:{cam.port}/SeCrEtKeY"
            with pytest.raises(RtspError) as err:
                async with RtspReader(secret):
                    pass
            return str(err.value)

    said = asyncio.run(go())
    assert "SeCrEtKeY" not in said and "hunter2" not in said and "404" in said
