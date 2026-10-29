import numpy as np

from movable_reality.sam import Instance, paste_mask


def test_paste_mask_box_local():
    local = np.zeros((2, 3), dtype=bool)
    local[0, 1] = True
    inst = Instance(object_id="0", frame_index=0, box=(4, 5, 7, 7), mask=local, confidence=None)
    full = paste_mask(inst, width=10, height=8)
    assert full.shape == (8, 10)
    assert full.sum() == 1
    assert full[5, 5]


def test_paste_mask_clips_to_frame():
    local = np.ones((4, 4), dtype=bool)
    inst = Instance(object_id="0", frame_index=0, box=(8, 6, 12, 10), mask=local, confidence=None)
    full = paste_mask(inst, width=10, height=8)
    assert full.sum() == 2 * 2


def test_paste_mask_resizes_when_raster_differs_from_box():
    local = np.ones((2, 2), dtype=bool)
    inst = Instance(object_id="0", frame_index=0, box=(0, 0, 4, 4), mask=local, confidence=None)
    assert paste_mask(inst, width=4, height=4).all()
