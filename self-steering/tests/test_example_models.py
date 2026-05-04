import argparse
import asyncio
import logging
import os
import time
import warnings
from typing import List

import llamppl
import pytest

from disciple.pipelines import SMCInferencePipeline
from evaluations.dataset import load_dataset
from evaluations.model_registry import ModelRegistry

logger = logging.getLogger(__name__)
warnings.simplefilter("always", UserWarning)


async def run_tests(
    task_types: List[str] = None,
    device_id: int = 0,
    huggingface_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    backend: str = "vllm",
    particles: int = 1,
    max_tokens: int = 128,
    ess_threshold: float = 0.5,
    temperature: float = 0.7,
    timeout: float = 120,
    clear_cache: bool = True,
):
    # Set the CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Backend:", backend)
    print("Particles:", particles)
    print("Max Tokens:", max_tokens)
    print("ESS Threshold:", ess_threshold)
    print("Temperature:", temperature)
    print("Timeout:", timeout)
    print("Clear Cache:", clear_cache)

    print(f"Loading {huggingface_model}...")
    GLOBAL_LM = llamppl.CachedCausalLM.from_pretrained(
        huggingface_model,
        backend=backend,
    )
    GLOBAL_LM.batch_size = particles
    pipeline = SMCInferencePipeline(results_dir=None)

    registry = ModelRegistry()
    datasets = {}
    summary_report = []

    for classname, metadata in registry.registry.items():
        if metadata["dataset"] == "examples":
            continue
        if task_types is not None and metadata["task_type"] not in task_types:
            continue

        print(f"Testing model {classname}...")

        # Load the dataset if it hasn't been loaded yet
        dataset_name = metadata["dataset"]
        if dataset_name not in datasets:
            datasets[dataset_name] = load_dataset(dataset_name)
        dataset = datasets[dataset_name]

        # Get the task associated with the model
        if metadata["task_id"] is not None:
            task = dataset.get_task_by_id(metadata["task_id"])
        else:
            # Puzzles dataset uses task_type instead of task_id
            task = dataset.get_tasks_by_type(metadata["task_type"])[0]
        assert task is not None
        assert metadata["task_type"] in task.task_types
        print(task)

        if clear_cache:
            GLOBAL_LM.clear_cache()
        start_time = time.time()
        task_results = await pipeline(
            task=task,
            model_cls=metadata["model_class"],
            N=particles,
            lm=GLOBAL_LM,
            max_tokens=max_tokens,
            temperature=temperature,
            ess_threshold=ess_threshold,
            timeout=timeout,
        )
        end_time = time.time()
        runtime = end_time - start_time

        all_passed = True
        for task_result in task_results:

            if task_result.error is not None:
                print(task_result.traceback)
                raise ValueError(
                    f"Error running model {classname}: {task_result.error}"
                )

            assert task_result.text is not None

            print(task_result.text)
            print(
                f"{'✅' if task_result.check_result else '❌'} `check()` -> {task_result.check_result}"
            )

            for evaluator_name, result in task_result.evaluate_results.items():
                print(f"{'✅' if result else '❌'} `{evaluator_name}` -> {result}")
                if task_result.check_result != result:
                    warnings.warn(f"⛔️ `check()` disagreed with `{evaluator_name}`")
                if not result:
                    all_passed = False

        if all_passed:
            print(f"✅ {classname} passed evaluation {evaluator_name}")
        else:
            warnings.warn(f"❌ {classname} failed evaluation {evaluator_name}")

        summary_report.append(
            {
                "classname": classname,
                "runtime": runtime,
                "all_passed": all_passed,
                "check_result": task_result.check_result,
            }
        )

    print("=" * 80)
    print("Summary Report:")
    for report in summary_report:
        if report["all_passed"] != report["check_result"]:
            status_emoji = "⛔️"
        elif report["all_passed"]:
            status_emoji = "✅"
        else:
            status_emoji = "❌"
        print(
            f"{status_emoji} Model: {report['classname']}, Runtime: {report['runtime']:.2f} seconds, All Passed: {report['all_passed']}, Check Result: {report['check_result']}"
        )
    print("=" * 80)


@pytest.mark.asyncio
@pytest.mark.gpu
async def test_example_models():
    await run_tests()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run example model tests with specific task types."
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=None,
        help="List of task types to run. If None, all task types will be run.",
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="The CUDA device ID to use."
    )
    parser.add_argument(
        "--huggingface-model",
        "-hf",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The HuggingFace model to use.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        help="The backend to use for CachedCausalLM.",
        choices=["vllm", "hf"],
    )
    parser.add_argument(
        "--particles",
        "-p",
        type=int,
        default=1,
        help="The number of particles.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="The maximum number of tokens in the generation.",
    )
    parser.add_argument(
        "--ess-threshold", type=float, default=0.5, help="The ESS threshold."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="LLM temperature."
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="Maximum inference time in seconds."
    )
    parser.add_argument(
        "--clear-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear the cache before running each model.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="The logging level.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    asyncio.run(
        run_tests(
            task_types=args.task_types,
            device_id=args.device_id,
            huggingface_model=args.huggingface_model,
            backend=args.backend,
            particles=args.particles,
            max_tokens=args.max_tokens,
            ess_threshold=args.ess_threshold,
            temperature=args.temperature,
            timeout=args.timeout,
            clear_cache=args.clear_cache,
        )
    )
