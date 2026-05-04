import ast
import datetime
import inspect
import json
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import load_index_from_storage
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import MessageRole
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from models import *
from openai import OpenAI
from tasks import TASKS
from tqdm import tqdm

load_dotenv()

PLACEHOLDER_CLASSNAME = "InferenceModel"
PLACEHOLDER_BASE_CLASSNAME = "BaseModel"
PLACEHOLDER_STEP_FUNCTION = "async def step(self):\n    pass\n"

DEFAULT_SYSTEM_PROMPT = (
    "You are writing Python code to solve a constrained string generation task using the `hfppl` library. "
    "You are given a partially-implemented `InferenceModel` class. "
    "Your job is to write the `step()` method, which is called iteratively to generate a string step-by-step ."
    "The `step()` method can return a single token, a word, or an entire sentence; you must decide the appropriate level of granularity for each task. "
)


class ModelGenerator:

    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

    # Output files
    DEFAULT_MODEL_DIR_BASE = "results_model_generation"

    # llamaindex
    LLAMAINDEX_CACHE_DIR = ".llamaindex"

    def __init__(
        self,
        openai_model=DEFAULT_OPENAI_MODEL,
        model_dir_base=DEFAULT_MODEL_DIR_BASE,
        llamaindex_cache_dir=LLAMAINDEX_CACHE_DIR,
    ):
        self.client = OpenAI()
        self.openai_model = openai_model
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.model_dir_base = model_dir_base
        self.llamaindex_cache_dir = llamaindex_cache_dir

    def get_ground_truth_model_source(
        self, task, exclude_step_method=False, obfuscate_classnames=True
    ):
        model_class = get_model_for_task(task)
        source = inspect.getsource(model_class)

        if obfuscate_classnames:
            source = source.replace(model_class.__name__, PLACEHOLDER_CLASSNAME)
            source = source.replace(
                model_class.__bases__[0].__name__, PLACEHOLDER_BASE_CLASSNAME
            )

        if exclude_step_method:
            method_source = inspect.getsource(getattr(model_class, "step"))
            source = source.replace(method_source, PLACEHOLDER_STEP_FUNCTION)

        return source

    def get_messages_for_task(self, task, system_prompt=DEFAULT_SYSTEM_PROMPT):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for task in [t for t in TASKS if t != task]:
            messages.append(
                {
                    "role": "user",
                    "content": self.get_ground_truth_model_source(
                        task, exclude_step_method=True
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.get_ground_truth_model_source(
                        task, exclude_step_method=False
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": self.get_ground_truth_model_source(
                    task, exclude_step_method=True
                ),
            }
        )
        return messages

    def compute_cost(self, response):
        if self.openai_model == "gpt-4o-mini":
            return (
                (response.usage.prompt_tokens * 0.15)
                + (response.usage.completion_tokens * 0.60)
            ) / 1e6
        elif self.openai_model == "gpt-4o":
            return (
                (response.usage.prompt_tokens * 5.00)
                + (response.usage.completion_tokens * 15.00)
            ) / 1e6

    def get_output_model_path(self, task: str):
        return os.path.join(
            self.model_dir_base,
            self.timestamp,
            self.openai_model,
            "models",
            f"{task}.py",
        )

    def get_output_results_path(self, task: str):
        return os.path.join(
            self.model_dir_base,
            self.timestamp,
            self.openai_model,
            "results",
            f"{task}.json",
        )

    def parsing_check(self, task):
        output_path_python = self.get_output_model_path(task)
        with open(output_path_python, "r") as f:
            source = f.read()
            try:
                ast.parse(source)
                return (True, None)
            except Exception as e:
                print(f"Parsing error for task {task}: {e}")
                return (False, str(e))

    def get_model_class(
        self,
        task,
        classname: str = PLACEHOLDER_CLASSNAME,
        base_classname: str = PLACEHOLDER_BASE_CLASSNAME,
    ):
        """Returns the generated model class for a given task."""
        output_path_python = self.get_output_model_path(task)
        if not os.path.exists(output_path_python):
            raise FileNotFoundError(
                f"Model file not found: {output_path_python}. Run `generate_model` first."
            )
        with open(output_path_python, "r") as f:
            source = f.read()
            source = source.replace(PLACEHOLDER_CLASSNAME, classname)
            source = source.replace(PLACEHOLDER_BASE_CLASSNAME, base_classname)
            exec(source, globals())
            return globals()[classname]

    def generate_model(self, task):
        messages = self.get_messages_for_task(task)
        response = self.client.chat.completions.create(
            model=self.openai_model, messages=messages
        )
        print(f"This query cost ${self.compute_cost(response):.4f}")

        output_path_python = self.get_output_model_path(task)
        os.makedirs(os.path.dirname(output_path_python), exist_ok=True)
        with open(output_path_python, "w") as f:
            f.write(response.choices[0].message.content)

        output_path_json = os.path.join(
            "results_model_generation",
            self.timestamp,
            self.openai_model,
            "responses",
            f"{task}.json",
        )
        os.makedirs(os.path.dirname(output_path_json), exist_ok=True)
        with open(output_path_json, "w") as f:
            f.write(response.model_dump_json(indent=4))

    def generate_model_with_llamaindex(
        self, task, similarity_top_k: int = 5, use_cached: bool = True
    ):
        if not os.path.exists(self.llamaindex_cache_dir) or not use_cached:
            documents = SimpleDirectoryReader(
                input_dir="../../hfppl",
                recursive=True,
                required_exts=[".md", ".py", ".ipynb"],
            ).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=self.llamaindex_cache_dir)
        else:
            storage_context = StorageContext.from_defaults(
                persist_dir=self.llamaindex_cache_dir
            )
            index = load_index_from_storage(storage_context)

        llm = LlamaOpenAI(model=self.openai_model)

        messages = self.get_messages_for_task(task)
        chat_history = [ChatMessage(**m) for m in messages[:-1]]
        chat_engine = index.as_chat_engine(
            llm=llm,
            chat_history=chat_history,
            chat_mode="context",
            similarity_top_k=similarity_top_k,
        )

        response = chat_engine.chat(message=messages[-1]["content"])

        output_path_python = self.get_output_model_path(task)
        os.makedirs(os.path.dirname(output_path_python), exist_ok=True)
        with open(output_path_python, "w") as f:
            f.write(response.response)

        parse_valid, parse_error = self.parsing_check(task)

        output_path_json = self.get_output_results_path(task)
        os.makedirs(os.path.dirname(output_path_json), exist_ok=True)

        with open(output_path_json, "w") as f:
            json.dump(
                {
                    "response": {
                        "source_nodes": [
                            {
                                "metadata": node.metadata,
                                "text": node.text,
                                "score": node.score,
                            }
                            for node in response.source_nodes
                        ]
                    },
                    "parse_valid": parse_valid,
                    "parse_error": parse_error,
                },
                f,
                indent=4,
            )
