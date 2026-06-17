import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HEIGHT = 8
WIDTH = 9


def rectangle_mask(x, y, w, h):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8, order="F")
    mask[y : y + h, x : x + w] = 1
    return mask


def mask_stack():
    first = rectangle_mask(3, 2, 4, 4)
    second = rectangle_mask(2, 1, 3, 3)
    return np.asfortranarray(np.stack([first, second], axis=2))


def uncompressed_counts(mask):
    counts = []
    value = 0
    run = 0
    for pixel in mask.reshape(-1, order="F"):
        if pixel != value:
            counts.append(run)
            run = 0
            value = pixel
        run += 1
    counts.append(run)
    return counts


@pytest.fixture(scope="module")
def mask_utils():
    pytest.importorskip("pyvistools._mask")
    from pyvistools import mask

    return mask


def test_encode_decode_round_trips_2d_and_3d_masks(mask_utils):
    mask = rectangle_mask(3, 2, 4, 4)

    rle = mask_utils.encode(mask)
    assert rle["size"] == [HEIGHT, WIDTH]
    assert np.array_equal(mask_utils.decode(rle), mask)

    stack = mask_stack()
    rles = mask_utils.encode(stack)
    decoded = mask_utils.decode(rles)

    assert [rle["size"] for rle in rles] == [[HEIGHT, WIDTH], [HEIGHT, WIDTH]]
    assert decoded.flags.f_contiguous
    assert np.array_equal(decoded, stack)


def test_area_bbox_and_merge_for_encoded_masks(mask_utils):
    stack = mask_stack()
    first = stack[:, :, 0]
    second = stack[:, :, 1]
    rles = mask_utils.encode(stack)

    assert mask_utils.area(rles).tolist() == [16, 9]
    assert mask_utils.toBbox(rles).astype(int).tolist() == [
        [3, 2, 4, 4],
        [2, 1, 3, 3],
    ]

    union = mask_utils.decode(mask_utils.merge(rles, intersect=0))
    intersection = mask_utils.decode(mask_utils.merge(rles, intersect=1))

    assert np.array_equal(union, np.maximum(first, second))
    assert np.array_equal(intersection, np.minimum(first, second))


def test_iou_matches_between_rle_and_bbox_inputs(mask_utils):
    rles = mask_utils.encode(mask_stack())
    boxes = np.array([[3.0, 2.0, 4.0, 4.0], [2.0, 1.0, 3.0, 3.0]])
    expected = np.array([[1.0, 4.0 / 21.0], [4.0 / 21.0, 1.0]])

    assert np.allclose(mask_utils.iou(rles, rles, [0, 0]), expected)
    assert np.allclose(mask_utils.iou(boxes, boxes, [0, 0]), expected)


def test_fr_py_objects_handles_bbox_polygon_and_uncompressed_rle(mask_utils):
    bbox_rles = mask_utils.frPyObjects(
        np.array([[1.0, 2.0, 3.0, 4.0]]),
        HEIGHT,
        WIDTH,
    )
    assert mask_utils.area(bbox_rles).tolist() == [12]
    assert mask_utils.toBbox(bbox_rles).astype(int).tolist() == [[1, 2, 3, 4]]
    assert np.array_equal(mask_utils.decode(bbox_rles)[:, :, 0], rectangle_mask(1, 2, 3, 4))

    polygon_rles = mask_utils.frPyObjects(
        [[2, 1, 6, 1, 6, 5, 2, 5]],
        HEIGHT,
        WIDTH,
    )
    assert mask_utils.area(polygon_rles).tolist() == [16]
    assert mask_utils.toBbox(polygon_rles).astype(int).tolist() == [[2, 1, 4, 4]]
    assert np.array_equal(mask_utils.decode(polygon_rles)[:, :, 0], rectangle_mask(2, 1, 4, 4))

    source = rectangle_mask(4, 0, 2, 3)
    uncompressed = [{"size": [HEIGHT, WIDTH], "counts": uncompressed_counts(source)}]
    uncompressed_rles = mask_utils.frPyObjects(uncompressed, HEIGHT, WIDTH)
    assert np.array_equal(mask_utils.decode(uncompressed_rles)[:, :, 0], source)


def test_cython_emits_gil_release_for_mask_api_calls(tmp_path):
    pytest.importorskip("Cython")

    pyx_path = ROOT / "pyvistools" / "_mask.pyx"
    generated_c = tmp_path / "_mask.c"
    subprocess.run(
        [sys.executable, "-m", "cython", "-3", str(pyx_path), "-o", str(generated_c)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    c_source = generated_c.read_text()

    for function_name in [
        "rleEncode",
        "rleDecode",
        "rleMerge",
        "rleArea",
        "rleIou",
        "bbIou",
        "rleToBbox",
        "rleFrBbox",
        "rleFrPoly",
        "rleToString",
        "rleFrString",
    ]:
        pattern = (
            r"_save = PyEval_SaveThread\(\);"
            r"(?:(?!PyEval_RestoreThread).)*"
            rf"\n\s+(?:__pyx_v_\w+\s*=\s*)?{function_name}\("
            r"(?:(?!PyEval_RestoreThread).)*"
            r"PyEval_RestoreThread\(_save\);"
        )
        assert re.search(pattern, c_source, flags=re.DOTALL), function_name
