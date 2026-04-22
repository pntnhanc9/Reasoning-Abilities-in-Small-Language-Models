import ast
import asyncio
import datetime
import inspect
import json
import linecache
import logging
import os
import re
import shutil
import string
import traceback
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Tuple

import jinja2
import llamppl
import nltk
from dotenv import load_dotenv
from openai import OpenAI

from disciple.base_models import BaseModel
from disciple.dev_models import DebugModel
from disciple.dev_models import UnconditionalSamplingModel
from disciple.masks import EOSMask
from disciple.masks import NewLineMask
from disciple.masks import NextTokenMask
from disciple.masks import PunctuationMask
from disciple.masks import TokenLengthMask
from evaluations.dataset import Task

PLACEHOLDER_CLASSNAME = "InferenceModel"
PLACEHOLDER_BASE_CLASSNAME = "BaseModel"

logger = logging.getLogger(__name__)


@dataclass
class ModelGenerationResult:
    model_path: str
    source: str
    cls: BaseModel
    reasoning: str = None

    # Error (optional)
    error: str = None
    traceback: str = None


class ModelGenerator(ABC):

    DEFAULT_MODEL_DIR_BASE = "model_generator"

    CACHE_BEHAVIOR_FORCE = "force"
    CACHE_BEHAVIOR_OPTIONAL = "optional"
    CACHE_BEHAVIOR_REQUIRE = "require"
    CACHE_BEHAVIOR_LATEST_READ_ONLY = "latest_read_only"

    def __init__(
        self,
        model_dir: str = None,
        cache_behavior: str = CACHE_BEHAVIOR_OPTIONAL,
        generate_step_fn: bool = True,
        generate_check_fn: bool = True,
        debug_mode: bool = False,
    ):
        """Base class for model generators.

        Args:
            model_dir: Directory to save generated models.
            cache_behavior: Determines how to handle caching.
                Options are:
                - 'force': Always generate new models, overriding any cached data.
                - 'optional': Use cached data if available, otherwise generate new models.
                - 'require': Require that cached data is available, otherwise raise an error.
                - 'latest_read_only': Like 'require', but only read from the latest version.
            generate_step_fn: If True, the generated model will include a step function. If False, the step function will be replaced with an unconditional step function.
            generate_check_fn: If True, the generated model will include a check function. NOTE: The behavior for False is not yet implemented.
            debug_mode: If True, calls to the ModelGenerator will produce the BaseModel source code.

        """
        self.generate_step_fn = generate_step_fn
        self.generate_check_fn = generate_check_fn

        if not self.generate_check_fn:
            # NOTE: This feature is implementable but requires additional work to ensure:
            # (1) mentions of check() are omitted from prompts
            # (2) check() is omitted from both BaseModel and all example models
            # (3) smc_standard() does not call check()
            raise NotImplementedError("generate_check_fn=False is not yet implemented.")

        allowed_cache_behaviors = {
            self.CACHE_BEHAVIOR_FORCE,
            self.CACHE_BEHAVIOR_OPTIONAL,
            self.CACHE_BEHAVIOR_REQUIRE,
            self.CACHE_BEHAVIOR_LATEST_READ_ONLY,
        }
        if cache_behavior is None:
            cache_behavior = self.CACHE_BEHAVIOR_OPTIONAL
        if cache_behavior not in allowed_cache_behaviors:
            raise ValueError(
                f"Invalid cache_behavior. Choose from {allowed_cache_behaviors}."
            )

        self.model_dir = (
            model_dir if model_dir is not None else self.DEFAULT_MODEL_DIR_BASE
        )
        self.cache_behavior = (
            cache_behavior
            if cache_behavior is not None
            else self.CACHE_BEHAVIOR_OPTIONAL
        )
        self.debug_mode = debug_mode if debug_mode is not None else False

        self.load_namespace()
        self.NAMESPACE_SOURCE = "\n".join(
            ["- " + library for library in self.local_namespace.keys()]
        )

        # Source code for BaseModel
        self.BASE_MODEL_SOURCE = self.get_source_from_class(
            BaseModel, obfuscate_classnames=False
        )

        # Source code for masks
        self.MASKS_SOURCE = "\n".join(
            [
                self.get_source_from_class(mask_class, obfuscate_classnames=False)
                for mask_class in [
                    NextTokenMask,
                    PunctuationMask,
                    EOSMask,
                    TokenLengthMask,
                    NewLineMask,
                ]
            ]
        )

    def load_namespace(self):
        # All imports required by the generated code should be included here
        self.local_namespace = {
            # Standard library
            "asyncio": asyncio,
            "datetime": datetime,
            "re": re,
            "string": string,
            # Third-party libraries
            "nltk": nltk,
            "llamppl": llamppl,
            # Custom classes
            "BaseModel": BaseModel,
            "NextTokenMask": NextTokenMask,
            "PunctuationMask": PunctuationMask,
            "EOSMask": EOSMask,
            "TokenLengthMask": TokenLengthMask,
            "NewLineMask": NewLineMask,
        }

    @staticmethod
    @abstractmethod
    def load_from(existing_model_dir: str, new_model_dir: str, **kwargs):
        pass

    def __call__(
        self, task: Task, example_models: List[Dict], version: int = 1, **kwargs
    ) -> ModelGenerationResult:
        # NOTE: Loading from cache is currently version-specific.
        if self.cache_behavior == self.CACHE_BEHAVIOR_OPTIONAL:
            if os.path.exists(self.get_model_path(task, version)):
                return self.load_model_class(task, version)
        elif self.cache_behavior == self.CACHE_BEHAVIOR_REQUIRE:
            return self.load_model_class(task, version)
        elif self.cache_behavior == self.CACHE_BEHAVIOR_LATEST_READ_ONLY:
            version = self.get_latest_version(task)
            return self.load_model_class(task, version)
        elif self.cache_behavior == self.CACHE_BEHAVIOR_FORCE:
            pass

        if self.debug_mode:
            return self.generate_model_debug_mode(
                task, example_models=example_models, version=version, **kwargs
            )

        return self.generate_model(
            task, example_models=example_models, version=version, **kwargs
        )

    @abstractmethod
    def generate_model(
        self, task: Task, example_models: List[Dict], version: int = 1, **kwargs
    ) -> ModelGenerationResult:
        pass

    def generate_model_debug_mode(
        self, task: Task, version: int = 1, **kwargs
    ) -> ModelGenerationResult:
        """Generates a BaseModel with the source code for debugging purposes."""
        model_path = self.get_model_path(task, version)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            f.write(self.get_source_from_class(DebugModel, obfuscate_classnames=True))

        return self.load_model_class(task, version)

    def get_latest_version(self, task: Task) -> int:
        """
        Returns the highest version number for the given task.
        """
        task_dir = os.path.join(self.model_dir, f"task_{task.task_id:04d}")
        if not os.path.exists(task_dir):
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        versions = []
        for entry in os.listdir(task_dir):
            entry_path = os.path.join(task_dir, entry)
            if os.path.isdir(entry_path) and entry.startswith("v"):
                version = int(entry[1:])
                versions.append(version)
        if not versions:
            raise FileNotFoundError(f"No versions found for task {task.task}")
        return max(versions)

    def get_model_path(self, task: Task, version: int = 1):
        return os.path.join(
            self.model_dir,
            f"task_{task.task_id:04d}",
            f"v{version:02d}",
            f"model.py",
        )

    def get_prompt_path(self, task: Task, version: int = 1):
        return os.path.join(
            self.model_dir,
            f"task_{task.task_id:04d}",
            f"v{version:02d}",
            f"prompt.json",
        )

    def get_completions_path(self, task: Task, version: int = 1):
        return os.path.join(
            self.model_dir,
            f"task_{task.task_id:04d}",
            f"v{version:02d}",
            f"completion.json",
        )

    def get_source_from_class(self, model_class, obfuscate_classnames=True):
        source = inspect.getsource(model_class)

        source = self._strip_decorators(source)

        if obfuscate_classnames:
            source = source.replace(model_class.__name__, PLACEHOLDER_CLASSNAME)
            source = source.replace(
                model_class.__bases__[0].__name__, PLACEHOLDER_BASE_CLASSNAME
            )

        return source

    def load_model_class(
        self,
        task: Task,
        version: int = 1,
        classname: str = PLACEHOLDER_CLASSNAME,
        base_classname: str = PLACEHOLDER_BASE_CLASSNAME,
    ) -> ModelGenerationResult:
        """Loads a generated model for a given task."""
        model_path = self.get_model_path(task, version)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Run `generate_model` first."
            )
        with open(model_path, "r") as f:
            source = f.read()

        source = source.replace(PLACEHOLDER_CLASSNAME, classname)
        source = source.replace(PLACEHOLDER_BASE_CLASSNAME, base_classname)

        if not self.generate_step_fn:
            source = self._replace_step_fn(
                original_source=source,
                target_source=inspect.getsource(UnconditionalSamplingModel),
            )

        # Filename displayed in tracebacks that result from execution errors
        filename = "inference_model.py"
        try:
            compiled_code = compile(source, filename, "exec")
        except Exception as e:
            return ModelGenerationResult(
                model_path=model_path,
                source=source,
                cls=None,
                error=str(e),
                traceback=traceback.format_exc(),
            )

        # Inject the generated code into linecache so tracebacks can display it
        linecache.cache["inference_model.py"] = (
            len(source),  # size of the code
            None,  # timestamp, can be None
            source.splitlines(keepends=True),  # list of source lines
            filename,
        )

        try:
            exec(compiled_code, self.local_namespace)
            model_cls = self.local_namespace[classname]
        except Exception as e:
            return ModelGenerationResult(
                model_path=model_path,
                source=source,
                cls=None,
                error=str(e),
                traceback=traceback.format_exc(),
            )

        return ModelGenerationResult(
            model_path=model_path, source=source, cls=model_cls
        )

    def parsing_check(self, task: Task, version: int = 1):
        model_path = self.get_model_path(task, version)
        with open(model_path, "r") as f:
            source = f.read()
            try:
                ast.parse(source)
                return (True, None)
            except Exception as e:
                logger.error(f"Parsing error for task {task}: {e}")
                return (False, str(e))

    @staticmethod
    def _replace_step_fn(original_source: str, target_source: str) -> str:
        """Replaces the step() function in the original source with the step() function from the target source."""

        target_tree = ast.parse(target_source)
        step_ast = None

        for node in ast.walk(target_tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "step":
                step_ast = node
                break

        if step_ast is None:
            raise ValueError("No step() function found in target source.")

        tree = ast.parse(original_source)
        node_found = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for i, body_node in enumerate(node.body):
                    if (
                        isinstance(body_node, ast.AsyncFunctionDef)
                        and body_node.name == "step"
                    ):
                        node.body[i] = step_ast
                        node_found = True
                        break

        if not node_found:
            raise ValueError("No step() function found in original source.")

        return ast.unparse(tree)

    @staticmethod
    def _strip_decorators(source: str) -> str:
        # Remove decorator lines that precede the class definition.
        lines = source.split("\n")
        new_lines = []
        class_found = False
        for line in lines:
            if line.strip().startswith("class "):
                class_found = True
            if class_found:
                new_lines.append(line)
        return "\n".join(new_lines)


class OpenAIModelGenerator(ModelGenerator):

    DEFAULT_OPENAI_MODEL = "gpt-4o-2024-08-06"
    SYSTEM_PROMPT_TEMPLATE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "prompts",
        "planner_system_prompt.md",
    )

    def __init__(
        self,
        openai_model: str = DEFAULT_OPENAI_MODEL,
        use_reasoning: bool = False,
        include_feedback: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.openai_model = openai_model

        load_dotenv()
        self.client = OpenAI()

        self.use_reasoning = use_reasoning
        self.include_feedback = include_feedback

        with open(self.SYSTEM_PROMPT_TEMPLATE, "r") as f:
            template = jinja2.Environment().from_string(f.read())
            self._system_prompt = template.render(
                BASE_MODEL_SOURCE=self.BASE_MODEL_SOURCE,
                MASKS_SOURCE=self.MASKS_SOURCE,
                NAMESPACE_SOURCE=self.NAMESPACE_SOURCE,
            )

    @staticmethod
    def load_from(existing_model_dir: str, new_model_dir: str, **kwargs):
        if not os.path.exists(existing_model_dir):
            raise FileNotFoundError(f"Directory not found: {existing_model_dir}")
        if os.path.exists(new_model_dir):
            raise FileExistsError(f"Directory already exists: {new_model_dir}")
        shutil.copytree(existing_model_dir, new_model_dir)
        logger.info(
            f"Loaded model generator from {existing_model_dir} to {new_model_dir}"
        )
        return OpenAIModelGenerator(model_dir=new_model_dir, **kwargs)

    def generate_model(
        self, task: Task, example_models: List[Dict], version: int = 1
    ) -> ModelGenerationResult:
        messages = self.get_messages_for_task(task, example_models)
        response = self.client.chat.completions.create(
            model=self.openai_model, messages=messages
        )
        response_text = response.choices[0].message.content
        logger.info(f"This query cost ${self.compute_cost(response):.4f}")

        # Extract code from response
        code, reasoning = self.extract_code(response_text)

        model_path = self.get_model_path(task, version)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            f.write(code)

        prompt_path = self.get_prompt_path(task, version)
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, "w") as f:
            f.write(json.dumps(messages, indent=4))

        completions_path = self.get_completions_path(task, version)
        os.makedirs(os.path.dirname(completions_path), exist_ok=True)
        with open(completions_path, "w") as f:
            f.write(response.model_dump_json(indent=4))

        result = self.load_model_class(task, version)
        result.reasoning = reasoning
        return result

    def get_messages_for_task(self, task: Task, example_models: List[Dict]):
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})
        for model_data in example_models:
            messages.append(
                {
                    "role": "user",
                    "content": model_data["prompt"],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_example_model(
                        model_data, use_reasoning=self.use_reasoning
                    ),
                }
            )
            if self.include_feedback:
                messages.append(
                    {
                        "role": "system",
                        "content": self.format_traceback(model_data.get("traceback")),
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": task.prompt,
            }
        )
        messages.append(
            {
                "role": "system",
                "content": self.format_constraint_prompt,
            }
        )
        return messages

    @property
    def system_prompt(self):
        return self._system_prompt

    @property
    def format_constraint_prompt(self):
        text = (
            f"Remember: Your goal is to implement an `InferenceModel` class that encodes the user's constraints. "
            f"Your output must contain a Python code block that defines a class named `InferenceModel`. "
        )

        if self.use_reasoning:
            text += f"You may use the space before the code block to reason about the problem. "
        else:
            text += f"Do not include any text outside of the code block. "

        return text

    @staticmethod
    def format_example_model(model_data: Dict, use_reasoning: bool = True):
        text = ""
        if use_reasoning and model_data.get("reasoning"):
            text += model_data.get("reasoning") + "\n\n"
        text += OpenAIModelGenerator.embed_code(model_data["model_source"])
        return text

    @staticmethod
    def format_traceback(traceback: str = None):
        if traceback is None:
            return "Your `InferenceModel` ran successfully!"
        else:
            text = f"Your `InferenceModel` encountered a runtime error:\n"
            text += f"```python\n{traceback}\n```"
            text += "Please fix the error and try again."
            return text

    @staticmethod
    def embed_code(code, start_text="```python", end_text="```"):
        return f"{start_text}\n{code}\n{end_text}"

    @staticmethod
    def extract_code(
        response_text, start_text="```python", end_text="```"
    ) -> Tuple[str, str]:
        """Extracts the first Markdown code block from response_text. If the code is not found, returns the original text. Also returns any reasoning text that precedes the code block."""
        start_idx = response_text.find(start_text)
        if start_idx == -1:
            return response_text
        end_idx = response_text.find(end_text, start_idx + len(start_text))
        if end_idx == -1:
            return response_text
        return (
            response_text[start_idx + len(start_text) : end_idx].strip(),
            response_text[:start_idx],
        )

    def compute_cost(self, response):
        if self.openai_model.startswith("gpt-4o-mini"):
            return (
                (response.usage.prompt_tokens * 0.15)
                + (response.usage.completion_tokens * 0.60)
            ) / 1e6
        elif self.openai_model.startswith("gpt-4o"):
            return (
                (response.usage.prompt_tokens * 2.50)
                + (response.usage.completion_tokens * 10.00)
            ) / 1e6
        elif self.openai_model.startswith("gpt-4.5"):
            return (
                (response.usage.prompt_tokens * 75.00)
                + (response.usage.completion_tokens * 150.00)
            ) / 1e6
        else:
            raise ValueError(f"Unknown model: {self.openai_model}")
