"""Smoke test for hard-negative-aware ObjectAwareRandomCrop."""
import numpy as np
import pytest

from animaloc.utils.augmentations import ObjectAwareRandomCrop


def _img(h=2048, w=2048):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_no_labels_field_falls_back_to_iguana_anchoring():
    """When labels are not passed, behaviour must be unchanged from pre-B."""
    crop = ObjectAwareRandomCrop(
        height=512, width=512,
        empty_probability=0.0,
        hard_negative_probability=0.5,  # would fire if H pool was found
    )
    img = _img()
    keypoints = [(1024.0, 1024.0)]
    params = crop.get_params_dependent_on_data({}, {
        "image": img, "keypoints": keypoints,
    })
    cx, cy = params["crop_x"], params["crop_y"]
    # Crop must contain the (single iguana) keypoint
    assert cx <= 1024 < cx + 512
    assert cy <= 1024 < cy + 512


def test_hard_negative_anchor_when_label_is_2():
    """With hard_negative_probability=1.0 and a label=2 keypoint, the crop
    must be anchored on the H point. Tested via the EXPLICIT `labels`
    side-channel (still supported even though albumentations v2 doesn't
    use it)."""
    crop = ObjectAwareRandomCrop(
        height=512, width=512,
        empty_probability=0.0,
        hard_negative_probability=1.0,
        hard_negative_label=2,
    )
    img = _img()
    keypoints = [
        (200.0, 200.0),    # iguana
        (1500.0, 1500.0),  # hard negative
    ]
    labels = [1, 2]
    for _ in range(20):
        params = crop.get_params_dependent_on_data({}, {
            "image": img, "keypoints": keypoints, "labels": labels,
        })
        cx, cy = params["crop_x"], params["crop_y"]
        assert cx <= 1500 < cx + 512, f"H point not in crop x: cx={cx}"
        assert cy <= 1500 < cy + 512, f"H point not in crop y: cy={cy}"


def test_hard_negative_anchor_via_packed_keypoint_tuples():
    """Albumentations v2 packs label_fields INTO keypoint tuples. With
    `hard_negative_probability=1.0` and an extended keypoint (x, y, 2),
    the crop must be anchored on it."""
    crop = ObjectAwareRandomCrop(
        height=512, width=512,
        empty_probability=0.0,
        hard_negative_probability=1.0,
        hard_negative_label=2,
    )
    img = _img()
    # Extended-tuple form: (x, y, label)
    keypoints = [
        (200.0, 200.0, 1),    # iguana
        (1500.0, 1500.0, 2),  # hard negative
    ]
    for _ in range(20):
        # No `labels` kwarg — must be read from keypoint[2].
        params = crop.get_params_dependent_on_data({}, {
            "image": img, "keypoints": keypoints,
        })
        cx, cy = params["crop_x"], params["crop_y"]
        assert cx <= 1500 < cx + 512, f"H point not in crop x: cx={cx}"
        assert cy <= 1500 < cy + 512, f"H point not in crop y: cy={cy}"


def test_iguana_anchor_when_no_hard_negative_present():
    """hard_negative_probability=0.8 but the keypoints list has no label=2
    entries → the H-anchored branch must NOT fire (otherwise we'd crop
    randomly on no anchors). Must fall back to iguana anchoring."""
    crop = ObjectAwareRandomCrop(
        height=512, width=512,
        empty_probability=0.0,
        hard_negative_probability=0.8,
        hard_negative_label=2,
    )
    img = _img()
    keypoints = [(1024.0, 1024.0)]
    labels = [1]  # no H
    # Run many iterations — crop must always anchor on the lone iguana
    for _ in range(20):
        params = crop.get_params_dependent_on_data({}, {
            "image": img, "keypoints": keypoints, "labels": labels,
        })
        cx, cy = params["crop_x"], params["crop_y"]
        assert cx <= 1024 < cx + 512
        assert cy <= 1024 < cy + 512


def test_empty_plus_hard_negative_probability_validates_sum():
    """Sum > 1.0 must be rejected."""
    with pytest.raises(ValueError, match="must be"):
        ObjectAwareRandomCrop(
            height=512, width=512,
            empty_probability=0.6,
            hard_negative_probability=0.5,
        )
