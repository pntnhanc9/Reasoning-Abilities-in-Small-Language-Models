import pytest

from evaluations.model_registry import ModelRegistry


def test_getitem():
    @ModelRegistry.register(task_type="task_type_1", dataset="dataset_1", task_id=0)
    class TestModel:
        pass

    registry = ModelRegistry()
    assert registry["TestModel"] == TestModel

    with pytest.raises(KeyError):
        registry["NonExistentModel"]


def test_get_models_by_task_type():
    @ModelRegistry.register(task_type="task_type_1", dataset="dataset1_1", task_id=0)
    class TestModel1:
        pass

    @ModelRegistry.register(task_type="task_type_2", dataset="dataset_2", task_id=0)
    class TestModel2:
        pass

    @ModelRegistry.register(task_type="task_type_1", dataset="dataset3", task_id=0)
    class TestModel3:
        pass

    registry = ModelRegistry()
    task_type_1_models = registry.get_models_by_task_type("task_type_1")
    assert TestModel1 in task_type_1_models
    assert TestModel3 in task_type_1_models
    assert TestModel2 not in task_type_1_models


def test_get_models_by_dataset():
    @ModelRegistry.register(task_type="task_type_1", dataset="dataset1_1", task_id=0)
    class TestModel1:
        pass

    @ModelRegistry.register(task_type="task_type_2", dataset="dataset_2", task_id=0)
    class TestModel2:
        pass

    @ModelRegistry.register(task_type="task_type_1", dataset="dataset1_1", task_id=1)
    class TestModel3:
        pass

    registry = ModelRegistry()
    dataset1_1_models = registry.get_models_by_dataset("dataset1_1")
    assert TestModel1 in dataset1_1_models
    assert TestModel3 in dataset1_1_models
    assert TestModel2 not in dataset1_1_models
