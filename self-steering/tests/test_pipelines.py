import asyncio
import json
import os
import shutil
from unittest.mock import mock_open
from unittest.mock import patch

import pytest

from disciple.pipelines import DisciplePipeline
from disciple.pipelines import load_pipelines_from_configs
from disciple.pipelines import pipeline_registry
from evaluations.dataset import Task


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path


@pytest.fixture
def config_file_test():
    return "test"


@pytest.fixture
def config_file_test_importance():
    return "test_importance"


@pytest.fixture
def config_file_test_rejection():
    return "test_rejection"


@pytest.fixture
def mock_invalid_config_file():
    with patch(
        "builtins.open",
        mock_open(read_data=json.dumps({"type": "unknown_pipeline", "params": {}})),
    ):
        yield


@pytest.fixture
def example_task():
    return Task(
        task_id=0,
        prompt="Please continue this sentence: the quick brown fox",
        evaluators={"valid": lambda x: True},
    )


def test_load_pipelines_from_valid_configs(config_file_test, results_dir):
    config_files = [config_file_test]
    pipelines = load_pipelines_from_configs(config_files, results_dir)
    assert len(pipelines) == 1

    pipeline = pipelines[0]
    assert isinstance(pipeline, DisciplePipeline)
    assert pipeline.model_generator.debug_mode == True


def test_load_pipelines_from_invalid_configs(mock_invalid_config_file, results_dir):
    config_files = ["invalid_config.json"]
    with pytest.raises(ValueError, match="Unknown pipeline type: unknown_pipeline"):
        load_pipelines_from_configs(config_files, results_dir)


def test_load_pipelines_from_missing_configs(results_dir):
    config_files = ["missing_config.json"]
    with pytest.raises(FileNotFoundError):
        load_pipelines_from_configs(config_files, results_dir)


@pytest.mark.asyncio
async def test_disciple_pipeline_generate_model(
    config_file_test, example_task, results_dir
):
    config_files = [config_file_test]
    pipelines = load_pipelines_from_configs(config_files, results_dir)

    assert len(pipelines) == 1
    pipeline = pipelines[0]

    model_generation_result = await pipeline.generate_model(
        task=example_task, example_models=[]
    )
    assert model_generation_result is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_file_strategy, expected_generate_step_fn, expected_source_equality",
    [
        ("config_file_test_importance", True, True),
        ("config_file_test_rejection", False, False),
    ],
)
async def test_disciple_sampling_pipeline_generate_model(
    request,
    config_file_test,
    config_file_strategy,
    expected_generate_step_fn,
    expected_source_equality,
    example_task,
    results_dir,
):
    """
    For importance sampling, expect the second pipeline to use generate_step_fn=True and have the same source as the first.
    For rejection sampling, expect generate_step_fn=False and a different source since step() is replaced.
    """
    # Retrieve the second config file fixture value
    config_second = request.getfixturevalue(config_file_strategy)
    config_files = [config_file_test, config_second]
    pipelines = load_pipelines_from_configs(config_files, results_dir)

    assert len(pipelines) == 2
    pipeline_1, pipeline_2 = pipelines
    assert pipeline_1.model_generator.model_dir == pipeline_2.model_generator.model_dir
    assert pipeline_2.model_generator.generate_step_fn == expected_generate_step_fn

    await pipeline_1.generate_model(task=example_task, example_models=[], version=1)
    model_generation_result_1 = await pipeline_1.generate_model(
        task=example_task, example_models=[], version=2
    )
    assert model_generation_result_1 is not None

    model_generation_result_2 = await pipeline_2.generate_model(
        task=example_task, example_models=[]
    )
    assert model_generation_result_2 is not None

    assert model_generation_result_1.model_path == model_generation_result_2.model_path
    assert "v02" in model_generation_result_1.model_path

    if expected_source_equality:
        assert model_generation_result_1.source == model_generation_result_2.source
    else:
        assert model_generation_result_1.source != model_generation_result_2.source


def test_load_pipelines_with_overlapping_names(config_file_test, results_dir):
    config_files = [config_file_test, config_file_test]
    with pytest.raises(ValueError, match="Duplicate pipeline name detected"):
        load_pipelines_from_configs(config_files, results_dir)
