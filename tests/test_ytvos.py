import pytest

from pyvistools.ytvos import YTVOS
from pyvistools.ytvoseval import YTVOSeval


def test_bbox_evaluation_from_memory_dataset():
    dataset = {
        'categories': [{'id': 1, 'name': 'object'}],
        'videos': [{'id': 1, 'width': 10, 'height': 10}],
        'annotations': [{
            'id': 1,
            'video_id': 1,
            'category_id': 1,
            'iscrowd': 0,
            'bboxes': [[0, 0, 10, 10]],
        }],
    }
    predictions = [{
        'video_id': 1,
        'category_id': 1,
        'score': 1.0,
        'bboxes': [[0, 0, 10, 10]],
    }]

    ytvos = YTVOS(dataset)
    ytvos_dets = ytvos.loadRes(predictions)
    evaluator = YTVOSeval(ytvos, ytvos_dets, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    assert ytvos_dets.anns[1]['areas'] == [100]
    assert ytvos_dets.anns[1]['avg_area'] == 100
    assert evaluator.stats[0] == pytest.approx(1.0)


def test_load_empty_results():
    ytvos = YTVOS({
        'categories': [{'id': 1, 'name': 'object'}],
        'videos': [{'id': 1, 'width': 10, 'height': 10}],
        'annotations': [],
    })

    results = ytvos.loadRes([])

    assert results.getAnnIds() == []
    assert results.getCatIds() == [1]
