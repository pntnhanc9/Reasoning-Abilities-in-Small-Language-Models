"""Script for running Collie benchmarks."""

import argparse
import asyncio
import datetime
import logging
import logging.config
import os
import sys
import time

import dill
import hfppl as hp
import numpy as np
import pandas as pd
from models import CollieModelRejectionSampling
from models import get_model_for_task
from tasks import TASKS
from tqdm import tqdm

DEFAULT_PARTICLES = [8, 16, 32, 64]

with open("hf_auth_token.txt", "r") as f:
    HF_AUTH_TOKEN = f.read().strip()


async def main(args):
    time_start = time.time()

    # Set seed
    rng = np.random.default_rng(args.seed)

    # Set CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # Set up results directory based on date and time
    results_dir = os.path.join(
        "results", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    os.makedirs(results_dir, exist_ok=True)

    # Save the logfile to the results directory
    logging.basicConfig(
        level=args.log_level,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, "log.txt")),
            logging.StreamHandler(),
        ],
    )

    # Save the args
    with open(os.path.join(results_dir, "args.txt"), "w") as f:
        f.write(str(args))

    GLOBAL_LM = hp.CachedCausalLM.from_pretrained(args.model, auth_token=HF_AUTH_TOKEN)

    with open("Collie/data/all_data.dill", "rb") as f:
        collie_dataset = dill.load(f)

    df_list = []

    for n_particles in args.particles:
        logging.info(f"Running with {n_particles} particles...")
        GLOBAL_LM.batch_size = n_particles

        for task in args.tasks:
            logging.info(f"Running task {task}...")
            dataset = collie_dataset[task]

            example_ids = np.arange(len(dataset))
            if args.n_examples is not None:
                example_ids = rng.choice(
                    example_ids, size=args.n_examples, replace=False
                )

            model_cls_list = [get_model_for_task(task)] + [CollieModelRejectionSampling]
            for model_cls in model_cls_list:
                logging.info(f"Running model {model_cls.__name__}...")

                # Clear cache
                GLOBAL_LM.clear_cache()

                for i, example_idx in tqdm(
                    enumerate(example_ids), total=len(example_ids)
                ):
                    example = dataset[example_idx]

                    prompt = get_prompt_for_example(
                        example,
                        GLOBAL_LM.tokenizer,
                        include_oneshot=args.include_oneshot,
                    )
                    logging.debug(prompt)

                    GLOBAL_LM.cache_kv(GLOBAL_LM.tokenizer.encode(prompt))

                    model = model_cls(
                        llm=GLOBAL_LM,
                        prompt=prompt,
                        constraints=example["constraint"],
                        targets=example["targets"],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )

                    try:
                        particles = await hp.smc_standard(
                            model,
                            n_particles=n_particles,
                            ess_threshold=args.ess_threshold,
                        )
                    except Exception as e:
                        logging.error(
                            f"Encountered an error while running model {model_cls.__name__} on task {task} (example_idx: {example_idx}): {e}",
                            exc_info=True,
                        )
                        continue

                    df = pd.DataFrame(
                        [
                            {
                                "task": task,
                                "example_n": i,
                                "example_idx": example_idx,
                                "model": model_cls.__name__,
                                "n_particles": n_particles,
                                "particle_id": n,
                                "valid": example["constraint"](
                                    str(p), target=example["targets"]
                                ),
                                "weight": p.weight,
                                "tokens": p.context.token_count,
                                "text": str(p),
                                "prompt": prompt,
                            }
                            for n, p in enumerate(particles)
                        ]
                    )
                    df_list.append(df)

                    # Save the results after every example
                    df = pd.concat(df_list).reset_index(drop=True)
                    df.to_csv(os.path.join(results_dir, "results.csv"), index=False)

    logging.info("Done!")
    logging.info(f"Total time: {time.time() - time_start:.2f} seconds")


def get_prompt_for_example(example, tokenizer, include_oneshot: bool = True):
    messages = [
        {
            "role": "system",
            "content": "You are helping a user generate text that satisfies constraints. Follow the user's instructions exactly. Write your response below; do not preface your response or include any additional remarks.",
        }
    ]
    if include_oneshot:
        oneshot_lines = example["oneshot_example"].split("\n")
        messages += [
            {"role": "user", "content": "\n".join(oneshot_lines[:-1])},
            {"role": "assistant", "content": oneshot_lines[-1]},
        ]

    messages += [{"role": "user", "content": example["prompt"]}]

    prompt = tokenizer.apply_chat_template(
        conversation=messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for running Collie benchmarks."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="The model to use.",
    )
    parser.add_argument(
        "--device-id", "-d", type=int, default=0, help="The CUDA device ID to use."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=TASKS,
        help="The tasks to run the benchmark on.",
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed.")
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="If set, subsample this many examples per task.",
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
        "--max-tokens", type=int, default=32, help="The maximum number of tokens."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="The temperature."
    )
    parser.add_argument(
        "--include-oneshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include oneshot example in the prompt.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="The logging level.",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()

    asyncio.run(main(args))
