"""Shared argument parser for benchmark evaluation scripts."""

import argparse
import os
from datetime import datetime

from disciple.model_generators import ModelGenerator

DEFAULT_PARTICLES = [1, 2, 4, 8, 16, 32]


def create_benchmark_parser():
    """Create the argument parser for benchmark evaluation scripts.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Script for running evaluation benchmarks."
    )

    ############################
    # General flags
    ############################
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="The results directory.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="The logging level.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--resume-from-experiment",
        type=str,
        default=None,
        help="The previous results directory to resume from.",
    )

    ############################
    # Dataset flags
    ############################
    parser.add_argument(
        "--dataset", type=str, default="collie", help="The dataset to run on."
    )
    parser.add_argument(
        "--task-types",
        type=str,
        nargs="+",
        default=None,
        help="The set of task types to run the benchmark on (defaults to all task types).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--n-examples-per-task-type",
        type=int,
        default=None,
        help="If set, subsample this many examples per task type.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="The number of times to repeat each task in the dataset (currently only applies to puzzle dataset).",
    )

    ############################
    # Pipeline flags
    ############################
    parser.add_argument(
        "--pipelines",
        type=str,
        nargs="+",
        help="One or more config.json files inside disciple/configs/.",
    )

    ############################
    # ModelGenerator flags
    ############################
    parser.add_argument(
        "--load-models-from-path",
        type=str,
        default=None,
        help="Load models from this directory. If `cache-behavior` is set to 'latest_read_only', expects a pipeline name in the same directory.",
    )
    parser.add_argument(
        "--cache-behavior",
        type=str,
        default=None,
        choices=[
            ModelGenerator.CACHE_BEHAVIOR_FORCE,
            ModelGenerator.CACHE_BEHAVIOR_OPTIONAL,
            ModelGenerator.CACHE_BEHAVIOR_REQUIRE,
            ModelGenerator.CACHE_BEHAVIOR_LATEST_READ_ONLY,
        ],
        help="Determines how to handle caching. Options are: 'force', 'optional', 'require', 'latest_read_only'.",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Whether to run ModelGenerator in debug mode.",
    )
    parser.add_argument(
        "--openai-model",
        "-o",
        type=str,
        default=None,
        help="The OpenAI model to use for the ModelGenerator.",
    )
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=None,
        help="The number of retries when generating a model. Defaults to 1.",
    )
    parser.add_argument(
        "--include-feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include feedback (i.e., tracebacks) in the model generation.",
    )

    ############################
    # Inference flags
    ############################
    parser.add_argument(
        "--huggingface-model",
        "-hf",
        type=str,
        default=None,
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
        "--device-id", type=int, default=0, help="The CUDA device ID to use."
    )
    parser.add_argument(
        "--particles",
        "-p",
        type=int,
        nargs="+",
        default=DEFAULT_PARTICLES,
        help="The number of particles.",
    )
    parser.add_argument(
        "--ess-threshold", type=float, default=0.5, help="The ESS threshold."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="The maximum number of tokens in the generation.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="LLM temperature."
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="Maximum inference time in seconds."
    )

    return parser


def parse_args_string(args_string):
    """Parse command line arguments string using the benchmark argument parser.

    Args:
        args_string: Command line arguments as a string from args.txt

    Returns:
        Dictionary with parsed arguments, or empty dict if parsing fails
    """
    import logging
    import shlex

    logger = logging.getLogger(__name__)

    if not args_string.strip():
        return {}

    try:
        # Split the args string into tokens, handling quoted arguments properly
        tokens = shlex.split(args_string)

        # Skip script name if present
        if tokens and tokens[0].endswith(".py"):
            tokens = tokens[1:]

        # Create the parser and parse the arguments
        parser = create_benchmark_parser()

        # Parse the arguments, handling unknown arguments gracefully
        try:
            args = parser.parse_args(tokens)
            # Convert Namespace to dict for easier use
            return vars(args)
        except SystemExit:
            # argparse calls sys.exit on error, catch this and return empty dict
            logger.warning(f"Failed to parse arguments: {args_string}")
            return {}

    except Exception as e:
        logger.warning(f"Error parsing arguments '{args_string}': {e}")
        return {}
