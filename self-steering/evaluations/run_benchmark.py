"""Script for running benchmarks."""

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from llamppl import CachedCausalLM
from tqdm import tqdm

from disciple.model_generators import ModelGenerator
from disciple.pipelines import InferenceResult
from disciple.pipelines import SamplingType
from disciple.pipelines import load_pipelines_from_configs
from evaluations.dataset import Task
from evaluations.dataset import load_dataset

DIR_RESULTS_BASE = "results"
DEFAULT_PARTICLES = [1, 2, 4, 8, 16, 32]

load_dotenv()

logger = logging.getLogger(__name__)


async def main(args):
    time_start = time.time()

    # Set up results directory
    if args.results_dir is None:
        args.results_dir = default_results_dir()
    if args.resume_from_experiment:
        # Copy the results directory
        shutil.copytree(
            args.resume_from_experiment, args.results_dir, dirs_exist_ok=True
        )
        with open(os.path.join(args.results_dir, "results.json"), "r") as f:
            results_list = []
            for result in json.load(f):
                task_kwargs = result["task"]
                task_kwargs["evaluators"] = task_kwargs.get("evaluators", [])
                result["task"] = Task(**task_kwargs)
                results_list.append(InferenceResult(**result))

        # Keep all results up to the highest task ID (remove incomplete results)
        highest_task_id = max(result.task.task_id for result in results_list)
        results_list = [
            result for result in results_list if result.task.task_id < highest_task_id
        ]
    else:
        # Create a new results directory
        os.makedirs(args.results_dir, exist_ok=True)
        results_list = []

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                os.path.join(
                    args.results_dir,
                    f"log.txt",
                ),
                mode="a" if args.resume_from_experiment else "w",
            ),
            logging.StreamHandler(),
        ],
    )

    if args.resume_from_experiment:
        logger.info(f"Resuming from {args.resume_from_experiment}")

    logger.info(f"Experiment : {os.path.basename(args.results_dir)}")
    logger.info(f"Results    : {args.results_dir}")

    # Set seed
    rng = np.random.default_rng(args.seed)

    # Set CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Save the command line arguments
    with open(os.path.join(args.results_dir, f"args.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Load the dataset
    dataset = load_dataset(args.dataset, task_types=args.task_types, repeat=args.repeat)
    task_ids = [task.task_id for task in dataset]

    # Optionally, resume from a specific example index
    if args.resume_from_experiment:
        if args.n_examples_per_task_type is not None:
            raise ValueError(
                "Cannot set both --resume-from-experiment and --n-examples-per-task-type simultaneously."
            )
        existing_task_ids = set([result.task.task_id for result in results_list])
        task_ids = [task_id for task_id in task_ids if task_id not in existing_task_ids]

    # Optionally, subsample examples
    if args.n_examples_per_task_type is not None:

        # Group tasks by task type
        task_type_to_task_ids = defaultdict(list)
        for task in dataset:
            for task_type in task.task_types:
                task_type_to_task_ids[task_type].append(task.task_id)

        # Subsample examples
        task_ids = []
        for task_type, task_ids_for_task_type in task_type_to_task_ids.items():
            task_ids.extend(
                rng.choice(
                    task_ids_for_task_type,
                    size=args.n_examples_per_task_type,
                    replace=True,
                )
            )

        # Deduplicate
        task_ids = list(set(task_ids))

    # Load the pipelines
    pipelines = load_pipelines_from_configs(
        config_files=args.pipelines,
        results_dir=args.results_dir,
        params=dict(
            dataset_name=dataset.name,
            model_generator_params=dict(
                openai_model=args.openai_model,
                load_models_from_path=args.load_models_from_path,
                cache_behavior=args.cache_behavior,
                include_feedback=args.include_feedback,
                debug_mode=args.debug_mode,
            ),
            n_attempts=args.n_attempts,
        ),
    )

    # Initialize the global language model
    if args.huggingface_model is not None:
        logger.info(
            f"Loading CachedCausalLM {args.huggingface_model} (backend={args.backend})..."
        )
        kwargs = {
            "model_id": args.huggingface_model,
            "backend": args.backend,
        }
        if args.backend == "vllm":
            kwargs["engine_opts"] = {
                "enable_chunked_prefill": False,
                "max_logprobs": 2
                * max(
                    args.particles
                ),  # vLLM requires 2x for beam search: https://github.com/vllm-project/vllm/issues/10792
            }
        GLOBAL_LM = CachedCausalLM.from_pretrained(**kwargs)
        patch_llama_chat_template(GLOBAL_LM.tokenizer)
    else:
        logger.warning("Skipped loading CachedCausalLM.")
        GLOBAL_LM = None

    task_ids = sorted(task_ids)

    for task_id in tqdm(
        task_ids,
        initial=len(existing_task_ids) if args.resume_from_experiment else 0,
        total=len(dataset) if args.n_examples_per_task_type is None else len(task_ids),
        desc="Tasks",
    ):
        task = dataset.get_task_by_id(task_id)
        logger.info(str(task))

        for pipeline in pipelines:

            logger.info(f"Running pipeline {pipeline}...")

            task_results = await pipeline(
                task=task,
                N=args.particles,
                lm=GLOBAL_LM,
                ess_threshold=args.ess_threshold,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )

            # Check that the number of final InferenceResult objects is correct
            n_final_results = len(
                list(filter(lambda x: x.is_final_attempt, task_results))
            )
            expected_results = {
                SamplingType.ADAPTIVE: sum(args.particles),
                SamplingType.INDEPENDENT: max(args.particles),
                SamplingType.SINGLE_SHOT: 1,
            }.get(pipeline.sampling_type)

            if n_final_results != expected_results:
                raise ValueError(
                    f"{pipeline.sampling_type} pipeline: Expected {expected_results} final results, but got {n_final_results}."
                )

            results_list.extend(task_results)

            save_results(results_list, os.path.join(args.results_dir, "results.json"))

    logger.info("Done!")
    logger.info(f"Total time: {time.time() - time_start:.2f} seconds")
    logger.info(f"Results saved to {args.results_dir}")


def save_results(results_list: List[InferenceResult], path: str):
    results_list = [
        dataclasses.asdict(
            dataclasses.replace(result, task=result.task.to_dict())
        )  # serialize Task object
        for result in results_list
    ]
    with open(path, "w") as f:
        json.dump(results_list, f, indent=4)


def default_results_dir():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        DIR_RESULTS_BASE,
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )


def patch_llama_chat_template(tokenizer):
    """Patches the chat template for the Llama 3.2 tokenizer. No-op for other tokenizers.

    Llama 3.2 tokenizer automatically injects today's current date into the chat template, which is not ideal for reproducibility.
    This function patches the chat template to use a fixed date instead.
    """
    if "Llama-3.2" in tokenizer.name_or_path:
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "llama_config",
                "llama-3.2-chat-template.jinja",
            ),
            "r",
        ) as f:
            template = f.read()
        tokenizer.chat_template = template


if __name__ == "__main__":
    from arg_parser import create_benchmark_parser

    parser = create_benchmark_parser()
    args = parser.parse_args()

    asyncio.run(main(args))
