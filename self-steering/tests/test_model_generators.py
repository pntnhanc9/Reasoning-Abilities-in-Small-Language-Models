import inspect
import os
import re
import shutil

import llamppl
import pytest

from disciple.base_models import BaseModel
from disciple.dev_models import UnconditionalSamplingModel
from disciple.model_generators import ModelGenerationResult
from disciple.model_generators import ModelGenerator
from disciple.model_generators import OpenAIModelGenerator
from evaluations.dataset import Task


@pytest.fixture
def model_generator(tmp_path):
    return OpenAIModelGenerator(
        model_dir=os.path.join(tmp_path, ModelGenerator.DEFAULT_MODEL_DIR_BASE),
        debug_mode=True,
    )


@pytest.fixture
def sample_task():
    return Task(
        task_id=1,
        prompt="Complete the sentence: the quick brown fox",
        evaluators={"test_evaluator": lambda x: True},
    )


def test_model_generator_initialization(model_generator):
    assert (
        model_generator.cache_behavior == OpenAIModelGenerator.CACHE_BEHAVIOR_OPTIONAL
    )


def test_openai_model_generator_initialization(model_generator):
    assert model_generator.openai_model == OpenAIModelGenerator.DEFAULT_OPENAI_MODEL


def test_load_from_existing_model_dir(tmp_path, model_generator, sample_task):
    existing_dir = model_generator.model_dir
    new_dir = os.path.join(tmp_path, "new_model_dir")

    # Generate a model into existing_dir
    result = model_generator(sample_task, example_models=[])
    assert isinstance(result, ModelGenerationResult)
    assert issubclass(result.cls, BaseModel)
    model_path = model_generator.get_model_path(sample_task)
    assert os.path.exists(model_path)

    # Load from existing_dir to new_dir
    new_model_generator = OpenAIModelGenerator.load_from(
        existing_model_dir=str(existing_dir),
        new_model_dir=str(new_dir),
        debug_mode=True,
    )
    assert os.path.exists(new_dir)

    # Assert that we can load the model class from new_dir
    loaded_model_cls = new_model_generator.load_model_class(sample_task)
    assert issubclass(loaded_model_cls.cls, BaseModel)


def test_generate_model(model_generator, sample_task):
    result = model_generator(sample_task, example_models=[])
    assert isinstance(result, ModelGenerationResult)
    assert issubclass(result.cls, BaseModel)
    model_path = model_generator.get_model_path(sample_task)
    assert os.path.exists(model_path)


def test_parsing_check(model_generator, sample_task):
    model_generator(sample_task, example_models=[])
    is_valid, error = model_generator.parsing_check(sample_task)
    assert is_valid
    assert error is None


def test_invalid_cache_behavior():
    with pytest.raises(ValueError):
        OpenAIModelGenerator(cache_behavior="invalid_behavior")


def test_cache_behavior_require_raises_error(model_generator, sample_task):
    model_generator.cache_behavior = OpenAIModelGenerator.CACHE_BEHAVIOR_REQUIRE
    with pytest.raises(FileNotFoundError):
        model_generator(sample_task, example_models=[])


def test_cache_behavior_optional_then_require(model_generator, sample_task):
    # First generate the model with OPTIONAL cache behavior
    model_generator.cache_behavior = OpenAIModelGenerator.CACHE_BEHAVIOR_OPTIONAL
    result = model_generator(sample_task, example_models=[])
    assert isinstance(result, ModelGenerationResult)
    assert issubclass(result.cls, BaseModel)
    model_path = model_generator.get_model_path(sample_task)
    assert os.path.exists(model_path)

    # Now switch to REQUIRE cache behavior and re-generate the model
    model_generator.cache_behavior = OpenAIModelGenerator.CACHE_BEHAVIOR_REQUIRE
    result = model_generator(sample_task, example_models=[])
    assert isinstance(result, ModelGenerationResult)
    assert issubclass(result.cls, BaseModel)
    assert os.path.exists(model_path)


def test_extract_code():
    response_text = """
    Here is the code you requested:
    ```python
    class InferenceModel(BaseModel):
        def step(self):
            return "example"
    ```
    """
    expected_code = """
    class InferenceModel(BaseModel):
        def step(self):
            return "example"
    """
    extracted_code = OpenAIModelGenerator.extract_code(response_text)[0]
    assert extracted_code.strip() == expected_code.strip()


def test_extract_code_no_code_block():
    response_text = "There is no code block in this response."
    extracted_code = OpenAIModelGenerator.extract_code(response_text)
    assert extracted_code == response_text


def test_extract_code_incomplete_code_block():
    response_text = """
    Here is the code you requested:
    ```python
    class InferenceModel(BaseModel):
        def step(self):
            return "example"
    """
    extracted_code = OpenAIModelGenerator.extract_code(response_text)
    assert extracted_code == response_text


def test_extract_code_multiple_code_blocks():
    response_text = """
    Here is the first code block:
    ```python
    class InferenceModel(BaseModel):
        def step(self):
            return "example"
    ```
    And here is the second code block:
    ```python
    class AnotherModel(BaseModel):
        def step(self):
            return "another example"
    ```
    """
    expected_code = """
    class InferenceModel(BaseModel):
        def step(self):
            return "example"
    """
    extracted_code = OpenAIModelGenerator.extract_code(response_text)[0]
    assert extracted_code.strip() == expected_code.strip()


def test_replace_step_fn_success():
    # Build a sample source containing an async def step block to be replaced.
    original_source = (
        "class Dummy:\n"
        "\n"
        "    def foo(self):\n"
        "        pass\n"
        "\n"
        "    async def step(self):\n"
        "        # original implementation\n"
        "        return 'dummy'\n"
        "\n"
        "    def bar(self):\n"
        "        pass\n"
    )

    target_source = inspect.getsource(UnconditionalSamplingModel)

    expected_source = (
        "class Dummy:\n"
        "\n"
        "    def foo(self):\n"
        "        pass\n"
        "\n"
        "    async def step(self):\n"
        '        """Each step samples an entire completion."""\n'
        "        while True:\n"
        "            token = await self.sample(self.context.next_token())\n"
        "            if token == self.EOS_TOKEN_ID:\n"
        "                self.finish()\n"
        "                return\n"
        "            if self.context.token_count > self.max_tokens:\n"
        "                self.condition(False)\n"
        "                return\n"
        "\n"
        "    def bar(self):\n"
        "        pass"
    )

    # Replace the async step block using _replace_step_fn
    new_source = ModelGenerator._replace_step_fn(original_source, target_source)

    # Check that the new source contains the unconditional step function code.
    assert new_source == expected_source


def test_replace_step_fn_no_match():
    # Build a sample source that does not contain an async def step block.
    original_source = (
        "class Dummy:\n"
        "\n"
        "    def foo(self):\n"
        "        pass\n"
        "\n"
        "    def step(self):\n"
        "        # not an async method\n"
        "        return 'dummy'\n"
    )
    target_source = inspect.getsource(UnconditionalSamplingModel)
    with pytest.raises(
        ValueError, match=re.escape("No step() function found in original source.")
    ):
        ModelGenerator._replace_step_fn(original_source, target_source)


def test_strip_decorators_no_decorator():
    source = "class MyClass:\n" "    def method(self):\n" "        pass\n"
    expected = source
    result = ModelGenerator._strip_decorators(source)
    assert result == expected


def test_strip_decorators_with_decorators():
    source = (
        "@decorator1\n"
        "@decorator2\n"
        "class MyClass:\n"
        "    def method(self):\n"
        "        pass\n"
    )
    expected = "class MyClass:\n" "    def method(self):\n" "        pass\n"
    result = ModelGenerator._strip_decorators(source)
    assert result == expected


def test_strip_decorators_mixed_whitespace():
    source = (
        "   @decorator\n"
        "   @another_decorator\n"
        "class MyClass:\n"
        "    def method(self):\n"
        "        pass\n"
    )
    # Only lines before the first line that starts with "class" (ignoring leading spaces) are removed.
    expected = "class MyClass:\n" "    def method(self):\n" "        pass\n"
    result = ModelGenerator._strip_decorators(source)
    assert result == expected


def test_strip_decorators_preserves_decorators_after_class():
    # Decorators that appear after the class definition should not be removed.
    source = (
        "class MyClass:\n" "    @staticmethod\n" "    def method():\n" "        pass\n"
    )
    expected = source
    result = ModelGenerator._strip_decorators(source)
    assert result == expected


def test_strip_decorators_multiline_decorator():
    source = (
        "@decorator_with_args(\n"
        "    'arg1',\n"
        "    'arg2'\n"
        ")\n"
        "@another_decorator\n"
        "class MyClass:\n"
        "    def method(self):\n"
        "        pass\n"
    )
    expected = "class MyClass:\n" "    def method(self):\n" "        pass\n"
    result = ModelGenerator._strip_decorators(source)
    assert result.strip() == expected.strip()
