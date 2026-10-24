import copy
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
UINT_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_uint))) - 1
SIZ_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_ulong))) - 1


@pytest.fixture(scope="module")
def mask_utils():
    pytest.importorskip("pyvistools._mask")
    from pyvistools import mask

    return mask


def counts_from_bitmap(bitmap):
    counts, value, run = [], 0, 0
    for pixel in bitmap.ravel(order="F"):
        if pixel != value:
            counts.append(run)
            value, run = pixel, 0
        run += 1
    return counts + [run]


def compressed_counts(counts):
    """Independent Python COCO serializer, also able to encode invalid counts."""
    encoded = bytearray()
    for i, count in enumerate(counts):
        value = count - counts[i - 2] if i > 2 else count
        while True:
            digit = value & 31
            value >>= 5
            more = value != (-1 if digit & 16 else 0)
            encoded.append((digit | (32 if more else 0)) + 48)
            if not more:
                break
    return bytes(encoded)


def make_rle(bitmap, encoding):
    counts = counts_from_bitmap(bitmap)
    if encoding == "bytes":
        counts = compressed_counts(counts)
    elif encoding == "str":
        counts = compressed_counts(counts).decode("ascii")
    elif encoding == "array":
        counts = np.array(counts, dtype=np.int64)
    elif encoding == "tuple":
        counts = tuple(counts)
    return {"size": list(bitmap.shape), "counts": counts}


def assert_pad_matches_bitmap(mask_utils, roi, size, offset, encoding):
    rle = make_rle(roi, encoding)
    original = copy.deepcopy(rle)
    result = mask_utils.pad(rle, size, offset)
    expected = np.zeros(size, dtype=np.uint8, order="F")
    x, y = offset
    expected[y : y + roi.shape[0], x : x + roi.shape[1]] = roi

    assert result["size"] == list(size)
    assert isinstance(result["counts"], bytes)
    # Comparing canonical runs also checks coalescing and removal of zero runs.
    assert result["counts"] == compressed_counts(counts_from_bitmap(expected))
    assert np.array_equal(mask_utils.decode(result), expected)
    assert result is not rle and result["size"] is not rle["size"]
    assert rle["size"] == original["size"]
    assert np.array_equal(rle["counts"], original["counts"])


@pytest.mark.parametrize("encoding", ["bytes", "str", "list", "tuple", "array"])
@pytest.mark.parametrize(
    "roi",
    [
        np.zeros((3, 4), dtype=np.uint8),
        np.ones((3, 4), dtype=np.uint8),
        # One foreground run starts/ends mid-column and crosses two boundaries.
        np.array([0, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8).reshape((3, 3), order="F"),
        np.indices((3, 4)).sum(axis=0).astype(np.uint8) % 2,
        np.array([[0, 1, 1, 0, 1]], dtype=np.uint8),
        np.array([[1], [0], [1]], dtype=np.uint8),
        np.zeros((0, 3), dtype=np.uint8),
        np.zeros((3, 0), dtype=np.uint8),
        np.zeros((0, 0), dtype=np.uint8),
    ],
)
def test_pad_deterministic(mask_utils, roi, encoding):
    h, w = roi.shape
    for size, offset in [
        ((h, w), (0, 0)),  # passthrough, including empty canvases
        ((h + 4, w + 5), (0, 0)),
        ((h + 4, w + 5), (5, 0)),
        ((h + 4, w + 5), (0, 4)),
        ((h + 4, w + 5), (5, 4)),  # exact bottom/right edges
        ((h + 4, w + 5), (2, 1)),
        ((h, w + 5), (2, 0)),  # no vertical padding: keep cross-column runs
    ]:
        assert_pad_matches_bitmap(mask_utils, roi, size, offset, encoding)


def test_pad_randomized(mask_utils):
    rng = np.random.default_rng(20261024)
    for _ in range(250):
        h, w = map(int, rng.integers(0, 10, size=2))
        dh, dw = map(int, rng.integers(0, 8, size=2))
        roi = (rng.random((h, w)) < rng.random()).astype(np.uint8)
        offset = (int(rng.integers(dw + 1)), int(rng.integers(dh + 1)))
        for encoding in ("list", "bytes", "str"):
            assert_pad_matches_bitmap(mask_utils, roi, (h + dh, w + dw), offset, encoding)


@pytest.mark.parametrize("counts", [[], [0], [0, 0, 0]])
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
def test_pad_empty_counts(mask_utils, counts, shape):
    for encoded in (counts, compressed_counts(counts), compressed_counts(counts).decode("ascii")):
        rle = {"size": shape, "counts": encoded}
        assert mask_utils.pad(rle, (3, 3), (0, 0)) == {"size": [3, 3], "counts": b"9"}


def test_pad_coalesces_runs_separated_by_zero_counts(mask_utils):
    for counts, expected in [([0, 2, 0, 4, 0], [0, 6]), ([2, 0, 4, 0], [6])]:
        for encoded in (counts, compressed_counts(counts)):
            result = mask_utils.pad({"size": [2, 3], "counts": encoded}, (2, 3), (0, 0))
            assert result["counts"] == compressed_counts(expected)


@pytest.mark.parametrize("size,offset", [((2, 3), (1, 0)), ((2, 3), (0, 1)),
    ((1, 4), (0, 0)), ((3, 2), (0, 0)), ((4, 5), (-1, 0)),
    ((4, 5), (0, -1)), ((4, 5), (6, 0)), ((4, 5), (0, 5)),
    ((-1, 5), (0, 0))])
def test_pad_rejects_out_of_bounds(mask_utils, size, offset):
    error = OverflowError if min(*size, *offset) < 0 else ValueError
    with pytest.raises(error):
        mask_utils.pad({"size": [2, 3], "counts": [6]}, size, offset)


@pytest.mark.parametrize("size,offset", [((4.0, 5), (0, 0)), ((4, 5), (0.5, 0)),
    ((4, 5), (0, "1"))])
def test_pad_rejects_noninteger_geometry(mask_utils, size, offset):
    with pytest.raises(TypeError):
        mask_utils.pad({"size": [2, 3], "counts": [6]}, size, offset)


def test_pad_accepts_numpy_integer_geometry(mask_utils):
    assert_pad_matches_bitmap(mask_utils, np.ones((2, 3), dtype=np.uint8),
        np.array([4, 5], dtype=np.int64), (np.int32(2), np.int64(1)), "array")


@pytest.mark.parametrize("rle,size,offset,error", [
    ([], (3, 3), (0, 0), TypeError),
    ({"size": [2, 3]}, (3, 3), (0, 0), KeyError),
    ({"size": [2, 3], "counts": [6]}, (3,), (0, 0), ValueError),
    ({"size": [2, 3], "counts": [6]}, (3, 3), (0, 0, 0), ValueError),
    ({"size": [-2, 3], "counts": []}, (3, 3), (0, 0), OverflowError),
    ({"size": [2.0, 3], "counts": [6]}, (3, 3), (0, 0), TypeError),
    ({"size": [0, 0], "counts": []}, (3, 3), (4, 0), ValueError),
    ({"size": [0, 4], "counts": []}, (3, 3), (0, 0), ValueError),
])
def test_pad_rejects_invalid_structure(mask_utils, rle, size, offset, error):
    with pytest.raises(error):
        mask_utils.pad(rle, size, offset)


@pytest.mark.parametrize("counts,error", [
    ([-1, 7], OverflowError), ([6.0], TypeError), (["6"], TypeError),
    ([UINT_MAX + 1], OverflowError), (None, TypeError),
    ([], ValueError), ([5], ValueError), ([7], ValueError),
    (b"5", ValueError), (b"7", ValueError), (b"", ValueError),
    (b"P", ValueError), (b"/", ValueError), (b"p", ValueError),
    ("\u00e9", ValueError),
    (b"o" * 20 + b"0", OverflowError),
    (compressed_counts([-1]), ValueError),
    (compressed_counts([UINT_MAX + 1]), OverflowError),
    (compressed_counts([0, UINT_MAX, 0, UINT_MAX + 1]), OverflowError),
])
def test_pad_rejects_invalid_counts(mask_utils, counts, error):
    with pytest.raises(error):
        mask_utils.pad({"size": [2, 3], "counts": counts}, (3, 4), (0, 0))


def test_pad_dimension_overflow(mask_utils):
    empty = {"size": [0, 0], "counts": []}
    for size, offset in [((SIZ_MAX + 1, 0), (0, 0)), ((2, SIZ_MAX), (0, 0)),
                         ((0, 0), (SIZ_MAX + 1, 0))]:
        with pytest.raises(OverflowError):
            mask_utils.pad(empty, size, offset)
    with pytest.raises(OverflowError):
        mask_utils.pad({"size": [SIZ_MAX, 2], "counts": []}, (0, 0), (0, 0))


@pytest.mark.parametrize("counts", [[UINT_MAX], [0, UINT_MAX]])
def test_pad_uint_max_count_round_trip(mask_utils, counts):
    for encoded in (counts, compressed_counts(counts), compressed_counts(counts).decode("ascii")):
        result = mask_utils.pad({"size": [1, UINT_MAX], "counts": encoded}, (1, UINT_MAX), (0, 0))
        assert result == {"size": [1, UINT_MAX], "counts": compressed_counts(counts)}


@pytest.mark.skipif(SIZ_MAX <= UINT_MAX, reason="requires 64-bit siz")
def test_pad_large_total_with_representable_runs(mask_utils):
    # Total area may exceed uint; every canonical run must still fit uint.
    counts = [0, UINT_MAX, 1, 1, UINT_MAX, 1]
    size = (1, sum(counts))
    for encoded in (counts, compressed_counts(counts)):
        assert mask_utils.pad({"size": size, "counts": encoded}, size, (0, 0)) == {
            "size": list(size), "counts": compressed_counts(counts)}


@pytest.mark.skipif(SIZ_MAX <= UINT_MAX, reason="requires 64-bit siz")
@pytest.mark.parametrize("rle,size,offset", [
    ({"size": [0, 0], "counts": []}, (1, UINT_MAX + 1), (0, 0)),
    ({"size": [1, 1], "counts": [0, 1]}, (1, UINT_MAX + 2), (0, 0)),
    ({"size": [1, 1], "counts": [0, 1]}, (1, UINT_MAX + 2), (UINT_MAX + 1, 0)),
    ({"size": [1, UINT_MAX + 1], "counts": [0, UINT_MAX, 0, 1]}, (1, UINT_MAX + 1), (0, 0)),
    ({"size": [1, UINT_MAX + 1], "counts": [UINT_MAX, 0, 1]}, (1, UINT_MAX + 1), (0, 0)),
])
def test_pad_rejects_output_count_overflow(mask_utils, rle, size, offset):
    for counts in (rle["counts"], compressed_counts(rle["counts"])):
        with pytest.raises(OverflowError):
            mask_utils.pad(dict(rle, counts=counts), size, offset)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses Linux address-space limits")
def test_pad_large_canvas_under_memory_limit(mask_utils):
    script = textwrap.dedent("""
        import os
        import resource
        from pyvistools import mask

        def forbidden(*args, **kwargs):
            raise AssertionError('pad must not encode or decode a bitmap')

        mask.encode = mask.decode = forbidden
        mask._mask.encode = mask._mask.decode = forbidden
        # After imports, allow only 64 MiB more virtual memory. A 60k-square
        # bitmap alone would require 3.6 GB, independent of physical paging.
        with open('/proc/self/statm') as f:
            current = int(f.read().split()[0]) * os.sysconf('SC_PAGE_SIZE')
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = current + 64 * 1024 * 1024
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        result = mask.pad({'size': [1, 1], 'counts': [0, 1]},
                          (60000, 60000), (30000, 30000))
        expected = mask.frPyObjects({'size': [60000, 60000],
            'counts': [1800030000, 1, 1799969999]}, 60000, 60000)
        assert result == expected
        # Wide foreground passthrough must not split at every source column.
        result = mask.pad({'size': [1, 1000000000], 'counts': [0, 1000000000]},
                          (1, 2000000000), (500000000, 0))
        expected = mask.frPyObjects({'size': [1, 2000000000],
            'counts': [500000000, 1000000000, 500000000]}, 1, 2000000000)
        assert result == expected
        # Skipping background must not allocate per ROI column either.
        assert mask.pad({'size': [1, 1000000000], 'counts': [1000000000]},
                        (2, 1000000000), (0, 1))['counts'] == mask.frPyObjects(
            {'size': [2, 1000000000], 'counts': [2000000000]}, 2, 1000000000)['counts']
    """)
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True,
                   capture_output=True, text=True, timeout=20,
                   env=dict(os.environ, OPENBLAS_NUM_THREADS="1"))
