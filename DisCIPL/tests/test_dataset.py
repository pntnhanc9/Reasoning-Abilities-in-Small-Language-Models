import warnings

import pytest

from evaluations.dataset import CollieDataset
from evaluations.dataset import IFEvalDataset
from evaluations.dataset import Task


def test_collie_dataset():
    dataset = CollieDataset()
    assert len(dataset) == 767
    # Check that all examples are valid
    for i, task in enumerate(dataset):
        assert task.task_id == i
        for example in task.examples:
            eval_results = task.evaluate(example)
            for name, result in eval_results.items():
                assert result is True


def test_collie_dataset_filtered():
    dataset = CollieDataset(task_types=["sent_01"])
    assert len(dataset) == 38


def test_ifeval_dataset():
    dataset = IFEvalDataset()
    assert len(dataset) == 541
    for i, task in enumerate(dataset):
        assert task.task_id == i


def test_ifeval_dataset_filtered():
    dataset = IFEvalDataset(
        task_types=["punctuation:no_comma", "length_constraints:number_words"]
    )
    assert len(dataset) == 30


def test_get_task_by_id_collie():
    dataset = CollieDataset()
    task = dataset.get_task_by_id(0)
    assert task.task_id == 0
    assert task.prompt is not None

    with pytest.raises(ValueError, match="No task found with task_id=999"):
        dataset.get_task_by_id(999)


def test_get_task_by_id_ifeval():
    dataset = IFEvalDataset()
    task = dataset.get_task_by_id(0)
    assert task.task_id == 0
    assert task.prompt is not None

    with pytest.raises(ValueError, match="No task found with task_id=999"):
        dataset.get_task_by_id(999)
