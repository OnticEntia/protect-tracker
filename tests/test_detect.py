import numpy as np

from tracker_service.detect import Detector, iou, resize


def raw_answer(w, h, things):
    """A YOLOX answer with the given things in it: (kind id, x1, y1, x2, y2, score) in pixels."""
    grid, stride = Detector.grids(w, h)
    out = np.zeros((len(grid), 85), np.float32)
    for cls, x1, y1, x2, y2, score in things:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        i = int(np.argmin(np.abs((grid[:, 0] + 0.5) * stride[:, 0] - cx) + np.abs((grid[:, 1] + 0.5) * stride[:, 0] - cy) + (stride[:, 0] != 16) * 1e6))
        s = stride[i, 0]
        out[i, :4] = (cx / s - grid[i, 0], cy / s - grid[i, 1], np.log((x2 - x1) / s), np.log((y2 - y1) / s))
        out[i, 4] = score
        out[i, 5 + cls] = 1.0
        out[i + 1, :] = out[i, :]  # the cell beside it says nearly the same: one box is kept
        out[i + 1, 0] -= 1
        out[i + 1, 4] = score * 0.9
    return out, grid, stride


def test_decode_sorts_by_kind_and_keeps_one_box_each():
    out, grid, stride = raw_answer(640, 384, [(0, 100, 50, 180, 300, 0.9), (2, 300, 200, 500, 320, 0.8), (7, 20, 300, 120, 350, 0.2)])
    found = Detector.decode(out, grid, stride, 640, 360, 0.35, ("person", "vehicle", "animal"))
    assert len(found["person"]) == 1 and len(found["vehicle"]) == 1 and found["animal"] == []
    x1, y1, x2, y2, score = found["person"][0]
    assert np.allclose((x1, y1, x2, y2), (100 / 640, 50 / 360, 180 / 640, 300 / 360), atol=1e-4) and abs(score - 0.9) < 1e-5
    assert np.allclose(found["vehicle"][0][:4], (300 / 640, 200 / 360, 500 / 640, 320 / 360), atol=1e-4)


def test_resize_keeps_things_where_they_are():
    big = np.zeros((720, 1280, 3), np.uint8)
    big[360:, 640:] = 200
    small = resize(big, 640, 360)
    assert small.shape == (360, 640, 3) and small[90, 160, 0] == 0 and small[270, 480, 0] == 200
    assert resize(big, 1280, 720) is big


def test_iou():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1 and iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    assert abs(iou((0, 0, 2, 2), (1, 0, 3, 2)) - 1 / 3) < 1e-9


def test_far_look_tiles_map_back_to_the_whole_picture():
    from tracker_service.camera import Engine, merge_far

    class OneBox:
        """A detector that finds a person in the middle of whatever it is shown."""

        def detect(self, tile, min_conf, kinds):
            assert tile.shape == (360, 640, 3) and min_conf == 0.6
            return {"person": [(0.4, 0.4, 0.6, 0.6, 0.9)]}

    engine = Engine.__new__(Engine)
    engine.detector, engine.size = OneBox(), (640, 360)

    class Frame:
        def to_ndarray(self, format, width, height):  # noqa: A002
            return np.zeros((height, width, 3), np.uint8)

    boxes = engine.far_people(Frame())
    assert len(boxes) == 4
    assert np.allclose(boxes[0][:4], (0.2, 0.2, 0.3, 0.3)) and np.allclose(boxes[3][:4], (0.7, 0.7, 0.8, 0.8))
    merged = merge_far([(0.19, 0.19, 0.31, 0.31, 0.8)], boxes)  # the first is one the whole-picture look had already
    assert len(merged) == 4 and merged[0][4] == 0.8
