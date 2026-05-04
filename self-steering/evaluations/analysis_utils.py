import json
import logging
import os

import numpy as np
import pandas as pd
from huggingface_hub import model_info

from disciple.pipelines import SamplingType
from evaluations.arg_parser import parse_args_string
from evaluations.dataset import load_dataset

logger = logging.getLogger(__name__)

ALL_PIPELINES = [
    "disciple_smc",
    "disciple_oracle_smc",
    "disciple_importance",
    "disciple_oracle_importance",
    "disciple_rejection",
    "disciple_oracle_rejection",
    "vllm",
    "vllm_beam_search",
    "huggingface",
    "huggingface_beam_search",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-cot",
    "o1",
]


def get_hf_model_details(hf_model: str) -> str:
    """Shorten the Hugging Face model name for display purposes."""
    if hf_model is None:
        return (None, None)
    model_name = hf_model.split("/")[-1]
    model_name = model_name.rstrip("-Instruct")
    model_family = model_name.split("-")[0]
    return model_name, model_family


def get_model_parameter_count(hf_model: str) -> int:
    """Get the parameter count of a Hugging Face model."""
    if hf_model is None:
        return None

    try:
        info = model_info(hf_model)
        return info.safetensors.get("total", 0) if info.safetensors else None
    except Exception as e:
        logger.warning(f"Could not get parameter count for {hf_model}: {e}")
        return None


def get_pipeline_info(pipeline_name, hf_model="Llama-3.2-1B"):
    hf_model_short, hf_model_family = get_hf_model_details(hf_model)

    PIPELINE_INFO = {
        "disciple_smc": {
            "name": "DisCIPL",
            "method": f"DisCIPL-SMC ({hf_model_short})",
            "method_short": "DisCIPL-SMC",
            "sampling_method": "SMC",
            "oracle": False,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "disciple_importance": {
            "name": "DisCIPL",
            "method": f"DisCIPL-IS ({hf_model_short})",
            "method_short": "DisCIPL-IS",
            "sampling_method": "Importance",
            "oracle": False,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "disciple_rejection": {
            "name": "DisCIPL",
            "method": f"DisCIPL-RS ({hf_model_short})",
            "method_short": "DisCIPL-RS",
            "sampling_method": "Rejection",
            "oracle": False,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "disciple_oracle_smc": {
            "name": "DisCIPL*",
            "method": f"DisCIPL*-SMC ({hf_model_short})",
            "method_short": "DisCIPL*-SMC",
            "sampling_method": "SMC",
            "oracle": True,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "disciple_oracle_importance": {
            "name": "DisCIPL*",
            "method": f"DisCIPL*-IS ({hf_model_short})",
            "method_short": "DisCIPL*-IS",
            "sampling_method": "Importance",
            "oracle": True,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "disciple_oracle_rejection": {
            "name": "DisCIPL*",
            "method": f"DisCIPL*-RS ({hf_model_short})",
            "method_short": "DisCIPL*-RS",
            "sampling_method": "Rejection",
            "oracle": True,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "vllm": {
            "name": "Follower-only",
            "method": hf_model_short,
            "method_short": "Sampling",
            "sampling_method": "Standard",
            "oracle": False,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "vllm_beam_search": {
            "name": "Follower-only",
            "method": f"{hf_model_short} (Beam)",
            "method_short": f"Beam Search",
            "sampling_method": "Beam Search",
            "oracle": False,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "huggingface": {
            "name": "Follower-only",
            "method": hf_model_short,
            "method_short": "Sampling",
            "sampling_method": "Standard",
            "oracle": False,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "huggingface_beam_search": {
            "name": "Follower-only",
            "method": f"{hf_model_short} (Beam)",
            "method_short": f"Beam Search",
            "sampling_method": "Beam Search",
            "oracle": False,
            "model": hf_model,
            "model_short": hf_model_short,
            "model_family": hf_model_family,
        },
        "gpt-4o-mini": {
            "name": "Planner-only",
            "method": "GPT-4o-mini",
            "method_short": "GPT-4o-mini",
            "sampling_method": "Standard",
            "oracle": False,
            "model": "GPT-4o-mini",
            "model_short": "GPT-4o-mini",
            "model_family": "GPT-4o-mini",
        },
        "gpt-4o": {
            "name": "Planner-only",
            "method": "GPT-4o",
            "method_short": "GPT-4o",
            "sampling_method": "Standard",
            "oracle": False,
            "model": "GPT-4o",
            "model_short": "GPT-4o",
            "model_family": "GPT-4o",
        },
        "gpt-4o-cot": {
            "name": "Planner-only",
            "method": "GPT-4o (CoT)",
            "method_short": "GPT-4o (CoT)",
            "sampling_method": "Chain-of-Thought",
            "oracle": False,
            "model": "GPT-4o",
            "model_short": "GPT-4o",
            "model_family": "GPT-4o",
        },
        "o1": {
            "name": "Reasoning",
            "method": "o1",
            "method_short": "o1",
            "sampling_method": "Standard",
            "oracle": False,
            "model": "o1",
            "model_short": "o1",
            "model_family": "o1",
        },
    }

    return PIPELINE_INFO[pipeline_name]


def is_disciple_pipeline(pipeline_name):
    return "disciple" in pipeline_name


def load_raw_data(timestamp_to_pipelines, core_cutoff="2025-04-01"):
    df_list = []
    existing_pipelines = set()

    for timestamp, pipelines in timestamp_to_pipelines.items():
        if pipelines is None:
            pipelines = ALL_PIPELINES  # default to all pipelines

        print(f"Loading results from {timestamp}: {pipelines}")
        _df = pd.read_json(f"results/{timestamp}/results.json")

        # Read experiment args from args.txt
        args_file_path = f"results/{timestamp}/args.txt"
        try:
            with open(args_file_path, "r") as f:
                experiment_args = f.read().strip()
        except FileNotFoundError:
            print(f"WARNING: args.txt not found for {timestamp}")
            experiment_args = ""

        # Add experiment_args as a new column
        experiment_args = parse_args_string(experiment_args)
        _df["experiment_args"] = [experiment_args] * len(_df)
        _df["hf_model"] = experiment_args.get("huggingface_model", None)

        _df["timestamp"] = timestamp
        _df["results_path"] = f"results/{timestamp}/results.json"
        _df["pipeline_path"] = f"results/{timestamp}/pipelines"

        # warn on duplicate pipelines
        new_pipelines = set(_df["pipeline_name"].unique())
        duplicates = new_pipelines.intersection(existing_pipelines)
        if duplicates:
            print(f"WARNING: {duplicates} already in existing pipelines")

        # filter the pipelines
        _df = _df[_df["pipeline_name"].isin(pipelines)]

        # mark timestamps before the core cutoff as "core"
        _df["core_result"] = pd.to_datetime(
            timestamp, format="%Y-%m-%d-%H-%M-%S"
        ) < pd.to_datetime(core_cutoff, format="%Y-%m-%d")

        df_list.append(_df)
    df = pd.concat(df_list).reset_index(drop=True)
    return df


def process_dataframe(df, dataset, rescore=False):
    # Unpack task dict into columns
    print("Unpacking task dict into columns")
    df_task = pd.json_normalize(df["task"])
    # Rename columns
    df_task = df_task.rename(columns={"prompt": "task_prompt"})
    # Drop columns
    if "evaluators" in df_task.columns:
        df_task = df_task.drop(columns=["evaluators"])
    if "examples" in df_task.columns:
        df_task = df_task.drop(columns=["examples"])

    # Unpack evaluate_results dict into columns
    print("Unpacking evaluate_results dict into columns")
    df_evaluate_results = pd.json_normalize(df["evaluate_results"])

    # Combine dataframes
    print("Combining dataframes")
    df = pd.concat([df_task, df, df_evaluate_results], axis=1)
    df = df.drop(columns=["task", "evaluate_results"])

    # Recompute "valid" column if rescore is True
    if rescore:
        assert "valid" in df.columns
        dataset = load_dataset(dataset, repeat=10)

        def _rescore(row):
            if not row["text"]:
                return None
            task_id = row["task_id"]
            task = dataset.get_task_by_id(task_id)
            return task.evaluate(row["text"])["valid"] == True

        df["valid"] = df.apply(_rescore, axis=1)

    if "valid" in df.columns:
        df["valid_true"] = df["valid"] == True
    elif "strict" in df.columns:
        df["valid_true"] = df["strict"] == True
    else:
        raise ValueError("No `valid` column found")

    if "error" not in df.columns:
        df["error"] = None
    df["error_true"] = ~df["error"].isnull()

    # Convert task_types (list) to string
    print("Converting task_types (list) to string")
    df["task_type"] = df["task_types"].apply(lambda x: ", ".join(x))

    # Extract task_type_n from task_type
    df["task_type_n"] = df["task_type"].str.extract(r"(\d+)").astype(float)

    # Map task_type to task_level
    print("Mapping task_type to task_level")

    def task_type_to_level(task_type):
        if "sent" in task_type:
            return "sentence"
        elif "para" in task_type:
            return "paragraph"
        else:
            return "other"

    df["task_level"] = df["task_type"].apply(task_type_to_level)

    # Add a column for method name based on pipeline_name and hf_model
    def get_method_name(row):
        pipeline_info = get_pipeline_info(row["pipeline_name"], row["hf_model"])
        return pipeline_info["method"]

    df["method"] = df[["pipeline_name", "hf_model"]].apply(get_method_name, axis=1)

    # Get exact parameter count for each Hugging Face model
    hf_model_parameters = {
        model: get_model_parameter_count(model) for model in df["hf_model"].unique()
    }
    df.loc[:, "hf_model_parameters"] = df["hf_model"].map(hf_model_parameters)

    df.loc[:, "hf_model_family"] = df["hf_model"].apply(
        lambda x: get_hf_model_details(x)[1] if x is not None else None
    )

    # Sort methods by order in ALL_PIPELINES with parameter count as a tiebreaker
    pipeline_order = {pipeline: i for i, pipeline in enumerate(ALL_PIPELINES)}

    def get_sort_key(method_row):
        pipeline_name = method_row.pipeline_name
        return (
            pipeline_order.get(pipeline_name, len(ALL_PIPELINES)),
            method_row.hf_model_family,
            method_row.hf_model_parameters,
        )

    unique_methods = df[
        ["pipeline_name", "method", "hf_model_family", "hf_model_parameters"]
    ].drop_duplicates()
    sorted_methods = sorted(unique_methods.itertuples(index=False), key=get_sort_key)
    method_order = [method for _, method, _, _ in sorted_methods]

    df["method"] = pd.Categorical(
        df["method"],
        categories=method_order,
        ordered=True,
    )

    # Reorder columns (put "task_id", "task_types", "task_prompt" first, then the rest)
    COL_ORDER = ["task_id", "task_types", "task_prompt", "pipeline_name", "method", "N"]
    COL_ORDER += [col for col in df.columns if col not in COL_ORDER]
    df = df[COL_ORDER]

    # Filter out pipelines that are not in the list
    # df = df[df["pipeline_name"].isin(PIPELINES.keys())].reset_index(drop=True)

    df["valid_breakdown"] = df[["valid_true", "error_true"]].apply(
        lambda x: (
            "Error" if x["error_true"] else ("Valid" if x["valid_true"] else "Invalid")
        ),
        axis=1,
    )

    # Add a column for the length of the text
    df["text_length"] = df["text"].apply(lambda x: len(x) if x is not None else 0)

    # Only include results from final attempts
    df_all_attempts = df.copy()
    df = df[df["is_final_attempt"] == True].reset_index(drop=True)

    return df, df_all_attempts


def calculate_disciple_token_usage(df):
    # Update token counts for DisCIPL pipelines
    print("Updating token counts for DisCIPL pipelines")
    # Use DataFrame apply to populate token usage in bulk
    mask = df["pipeline_name"].apply(is_disciple_pipeline)

    def _usage_series(row):
        usage = get_disciple_model_generator_token_usage(row)
        return pd.Series(
            {
                "reasoning_tokens": usage["completion_tokens"],
                "model_generator_prompt_tokens": usage["prompt_tokens"],
                "model_generator_prompt_tokens_cached": usage["prompt_tokens_details"][
                    "cached_tokens"
                ],
                "model_generator_completion_tokens": usage["completion_tokens"],
            }
        )

    # Apply only on DisCIPL rows and assign back to df
    token_df = df.loc[mask].apply(_usage_series, axis=1)
    df.loc[
        mask,
        [
            "reasoning_tokens",
            "model_generator_prompt_tokens",
            "model_generator_prompt_tokens_cached",
            "model_generator_completion_tokens",
        ],
    ] = token_df


def get_disciple_model_generator_token_usage(row):
    if not is_disciple_pipeline(row["pipeline_name"]):
        raise ValueError(f"Pipeline {row['pipeline_name']} is not a DisCIPL pipeline.")

    def get_completion_path(model_dir, task_id, version: int = 1):
        return os.path.join(
            model_dir,
            f"task_{task_id:04d}",
            f"v{version:02d}",
            f"completion.json",
        )

    load_models_from_pipeline = (
        "disciple_oracle_smc" if "oracle" in row["pipeline_name"] else "disciple_smc"
    )

    model_dir = os.path.join(
        row["pipeline_path"], load_models_from_pipeline, "model_generator"
    )
    completion_path = get_completion_path(
        model_dir, row["task_id"], version=row["attempt"]
    )

    with open(completion_path, "r") as f:
        completion = json.load(f)

    return completion["usage"]


def elementary_symmetric_sum(weights, k):
    """
    Compute the elementary symmetric sum of order k for the list of weights.
    That is, compute:
      E(k) = sum_{S ⊆ {1,...,n}, |S|=k} (∏_{i in S} weights[i])

    Uses a dynamic programming approach.
    """
    n = len(weights)
    E = [0.0] * (k + 1)
    E[0] = 1.0
    for a in weights:
        # Update backwards to ensure each weight is used only once.
        for j in range(k, 0, -1):
            E[j] += a * E[j - 1]
    return E[k]


def weighted_pass_at_k(passes, k, logprobs=None):
    """
    Compute the weighted pass@k metric.

    Parameters:
      passes: list of bools (length N)
          Indicator for each sample: True if the sample passes, False if it fails.
      k: int
          The number of samples drawn.
      logprobs: list of floats, optional (length N)
          If provided, these are interpreted as log-probabilities (which need not be normalized).
          If not provided, all samples are treated as having equal weight (i.e. weight 1.0).

    Returns:
      float: The weighted pass@k value:
             1 - (weighted sum over failing k-subsets)/(weighted sum over all k-subsets).

    Explanation:
      - The weights for each sample are computed as:
            weight = exp(logprob)   if logprobs is provided,
            weight = 1.0            otherwise.
      - The denominator is the weighted sum over all k-subsets.
      - The numerator is the weighted sum over k-subsets that contain only failing samples.
      - The ratio is the weighted probability that a k-subset contains no passing samples.
      - Subtracting from 1 gives the weighted pass@k.
    """
    N = len(passes)
    if k > N:
        raise ValueError("k cannot exceed the number of samples (N).")

    # If logprobs are provided, use them (via exponentiation); otherwise, use equal weights.
    if logprobs is not None:
        if len(logprobs) != N:
            raise ValueError("Length of logprobs must match length of passes.")
        weights = [np.exp(w) if not np.isnan(w) else 0.0 for w in logprobs]
    else:
        weights = [1.0] * N

    # Denom: weighted sum over all k-subsets.
    denom = elementary_symmetric_sum(weights, k)

    # Build weights for failing samples.
    fail_weights = [w for w, passed in zip(weights, passes) if not passed]
    if len(fail_weights) < k:
        # Not enough failing samples to form a k-subset implies that every k-subset
        # must contain at least one passing sample.
        return 1.0

    # Numer: weighted sum over k-subsets formed only from failing samples.
    numer = elementary_symmetric_sum(fail_weights, k)

    # Return the weighted pass@k metric.
    return 1.0 - numer / denom if denom != 0 else 0.0


def compute_pass_at_k(df, k_values, n_values, ablate_check_fn=False):
    """Computes weighted pass@k for each pipeline, task, N, and k value.

    Args:
        df: DataFrame containing the evaluation results.
        k_values: List of k values to compute pass@k for.
        n_values: List of N values to compute pass@k for.
        ablate_check_fn: If True, ablate the check function for pipelines that have it.

    Returns:
        DataFrame with the pass@k results.

    """
    df = df.copy()

    n_values = sorted(n_values)
    k_values = sorted(k_values)
    print(f"Computing pass@k for N={n_values}, k={k_values}")

    pass_at_k_results = []

    GROUP_COLS = ["pipeline_name", "method", "task_type", "task_id"]
    for (pipeline_name, method, task_type, task_id), data in df.groupby(GROUP_COLS)[
        df.columns.tolist()
    ]:

        sampling_type = data["sampling_type"].unique().item()
        class_weight = data["class_weight"].unique().item()
        hf_model = data["hf_model"].unique().item()
        hf_model_parameters = data["hf_model_parameters"].unique().item()

        # Whether the method defines a check() function
        has_check = is_disciple_pipeline(pipeline_name)

        for N in n_values:

            if sampling_type == SamplingType.ADAPTIVE:
                assert len(data) == sum(n_values)
                subset = data[data["N"] == N]
            elif sampling_type == SamplingType.INDEPENDENT:
                assert len(data) == max(n_values)
                assert (data["N"] == max(n_values)).all()
                subset = data.head(N)
            elif sampling_type == SamplingType.SINGLE_SHOT:
                assert len(data) == 1
                assert (data["N"] == 1).all()
                subset = pd.concat([data] * N, ignore_index=True)
            else:
                raise ValueError(f"Unknown sampling type: {sampling_type}")

            for k in k_values:
                if k > N:
                    break

                check_options = (
                    [False, True] if has_check and ablate_check_fn else [has_check]
                )
                for use_check in check_options:

                    # If use_check is True, set the weights to -inf for invalid samples.
                    if use_check:
                        weights = [
                            -np.inf if not check else w
                            for w, check in zip(
                                subset["weight"].tolist(),
                                subset["check_result"].tolist(),
                            )
                        ]

                    pass_at_k = weighted_pass_at_k(
                        passes=subset["valid_true"].tolist(),
                        k=k,
                        logprobs=(weights if "disciple" in pipeline_name else None),
                    )
                    error_true = subset["error_true"].all()

                    pass_at_k_results.append(
                        {
                            "pipeline_name": pipeline_name,
                            "method": method,
                            "hf_model": hf_model,
                            "hf_model_parameters": hf_model_parameters,
                            "sampling_type": sampling_type,
                            "has_check": has_check,
                            "use_check": use_check,
                            "task_type": task_type,
                            "class_weight": class_weight,
                            "task_id": task_id,
                            "N": N,
                            "k": k,
                            "pass@k": pass_at_k,
                            "error_true": error_true,
                        }
                    )

    df_pass_at_k = pd.DataFrame.from_records(pass_at_k_results)
    df_pass_at_k["method"] = pd.Categorical(
        df_pass_at_k["method"],
        categories=df["method"].cat.categories,
        ordered=True,
    )
    df_pass_at_k = df_pass_at_k.sort_values(
        ["method", "use_check", "task_type", "N", "k"]
    ).reset_index(drop=True)
    return df_pass_at_k


def compute_normalized_weights(logprobs: np.ndarray) -> np.ndarray:
    """Computes normalized weights from log-probabilities, guaranteed to sum to 1."""
    logprobs = np.array(logprobs)

    if np.isnan(logprobs).any():
        raise ValueError("Encountered NaN values in logprobs.")

    if (logprobs > 0).any():
        logger.warning("WARNING: Encountered positive values in logprobs.")

    if np.isneginf(logprobs).all():
        return np.ones_like(logprobs) / len(logprobs)

    probs = np.exp(logprobs - np.max(logprobs))
    return probs / np.sum(probs)


def sample_examples(
    df, N: int = None, use_check: bool = True, seed: int = 123
) -> pd.DataFrame:
    """Samples one example for each pipeline x task.

    Returns:
        df_samples: DataFrame with one example per pipeline x task.

    """
    rng = np.random.default_rng(seed)
    GROUP_COLS = ["pipeline_name", "method", "task_type", "task_id"]

    samples = []
    for (pipeline_name, method, task_type, task_id), data in df.groupby(GROUP_COLS)[
        df.columns.tolist()
    ]:

        sampling_type = data["sampling_type"].unique().item()
        nmax = data["N"].max()
        if N is None:
            N = nmax

        if sampling_type == SamplingType.ADAPTIVE:
            assert nmax == df["N"].max()
            subset = data[data["N"] == N]
        elif sampling_type == SamplingType.INDEPENDENT:
            assert nmax == df["N"].max()
            assert (data["N"] == nmax).all()
            subset = data.head(N)
        elif sampling_type == SamplingType.SINGLE_SHOT:
            assert nmax == 1
            assert (data["N"] == 1).all()
            subset = data
        else:
            raise ValueError(f"Unknown sampling type: {sampling_type}")

        # Apply check_fn to weights
        if is_disciple_pipeline(pipeline_name):
            # If all samples are errors, set all weights to -inf
            if subset["error_true"].all():
                logprobs = [-np.inf] * len(subset)
            # If use_check is True, set the weights to -inf for invalid samples
            elif use_check:
                logprobs = [
                    -np.inf if not check else w
                    for w, check in zip(
                        subset["weight"].tolist(), subset["check_result"].tolist()
                    )
                ]
            # Otherwise, use the original weights
            else:
                logprobs = subset["weight"].tolist()
        else:
            # Uniform weights
            logprobs = [0.0] * len(subset)
        weights = compute_normalized_weights(logprobs)
        selected = subset.sample(n=1, weights=weights, random_state=rng)
        samples.append(selected)

    df_samples = (
        pd.concat(samples)
        .sort_values(["method", "task_type", "task_id"])
        .reset_index(drop=True)
    )
    return df_samples


def generate_collie_table(df_pass_at_k):
    df_table = df_pass_at_k[df_pass_at_k["k"] == 1].copy()
    df_table["task_level"] = df_table["task_type"].apply(
        lambda task_type: "sentence" if "sent" in task_type else "para"
    )
    df_table = df_table[
        (
            (df_table["task_level"] == "sentence")
            & (
                df_table["N"]
                == df_table[df_table["task_level"] == "sentence"]["N"].max()
            )
        )
        | (
            (df_table["task_level"] == "paragraph")
            & (
                df_table["N"]
                == df_table[df_table["task_level"] == "sentence"]["N"].max()
            )
        )
    ]

    df_table = df_table.groupby(["method", "task_level"], observed=False)[
        ["pass@k", "error_true"]
    ].agg("mean")
    df_table = df_table.unstack("task_level")
    df_table = df_table.swaplevel(0, 1, axis=1)
    df_table = df_table.sort_index(axis=1, level=0)

    # Force the column order so that for each task_level, "pass@k" comes before "error_true"
    # Also, ensure that "sentence" appears left of "paragraph"
    desired_order = [
        ("sentence", "pass@k"),
        ("sentence", "error_true"),
        ("paragraph", "pass@k"),
        ("paragraph", "error_true"),
    ]
    df_table = df_table.reindex(columns=desired_order)

    df_table = df_table.rename(
        columns={"sentence": "Sentence", "paragraph": "Paragraph"}, level=0
    )
    df_table = df_table.rename(
        columns={"pass@k": "Pass@1", "error_true": "Error"}, level=1
    )

    # Reorder the rows to use the order from df_pass_at_k["method"]
    row_order = df_pass_at_k["method"].cat.categories.tolist()
    df_table = df_table.reindex(row_order)

    df_table = df_table.round(3)
    df_table.index.name = None

    return df_table


def generate_overall_results_table(
    df_pass_at_1,
    df_samples,
    class_weights,
    N_VALUES,
    round_digits=3,
    include_error: bool = True,
    pass_at_1_abbr: str = "Pass@1",
    coherency_abbr: str = "Coherency",
    error_abbr: str = "Error",
    include_method_info: bool = True,
    selected_task_types: list = None,  # Filter to only show these task types
    include_overall: bool = True,  # Option to exclude the "Overall" column
    transpose_table: bool = False,  # Option to transpose the table
    combine_metrics: bool = False,  # Combine metrics in one column with ± separator
    metrics_to_include: list = None,  # Only show these metrics
):
    # Filter Pass@1 data for max N
    df_pass_at_1_for_nmax = df_pass_at_1[df_pass_at_1["N"] == max(N_VALUES)]

    # Filter to selected task types if specified
    if selected_task_types:
        task_mask = df_pass_at_1_for_nmax["task_type"].isin(selected_task_types)
        df_pass_at_1_for_nmax = df_pass_at_1_for_nmax[task_mask]
        task_mask = df_samples["task_type"].isin(selected_task_types)
        df_samples = df_samples[task_mask]
        # Update class_weights to only include selected task types
        class_weights = {
            k: v for k, v in class_weights.items() if k in selected_task_types
        }

    # If metrics_to_include is specified, update what to include
    if metrics_to_include:
        include_error = error_abbr in metrics_to_include

    # Create pivot table for Pass@1 scores per task type
    pivot_pass = df_pass_at_1_for_nmax.pivot_table(
        index="method",
        columns="task_type",
        values="pass@k",
        aggfunc="mean",
        observed=False,
    )
    pivot_pass.columns = pd.MultiIndex.from_tuples(
        [(task, pass_at_1_abbr) for task in pivot_pass.columns]
    )

    # Create pivot table for Error scores per task type if requested
    if include_error:
        pivot_err = df_pass_at_1_for_nmax.pivot_table(
            index="method",
            columns="task_type",
            values="error_true",
            aggfunc="mean",
            observed=False,
        )
        pivot_err.columns = pd.MultiIndex.from_tuples(
            [(task, error_abbr) for task in pivot_err.columns]
        )
    else:
        pivot_err = None

    # Create pivot table for Coherency scores per task type
    if metrics_to_include is None or coherency_abbr in metrics_to_include:
        pivot_coh = df_samples.pivot_table(
            index="method",
            columns="task_type",
            values="coherency_score",
            aggfunc="mean",
            observed=False,
        )
        pivot_coh.columns = pd.MultiIndex.from_tuples(
            [(task, coherency_abbr) for task in pivot_coh.columns]
        )
    else:
        pivot_coh = None

    # Combine the pivot tables
    pivot_tables = []
    if metrics_to_include is None or pass_at_1_abbr in metrics_to_include:
        pivot_tables.append(pivot_pass)
    if pivot_coh is not None:
        pivot_tables.append(pivot_coh)
    if include_error and pivot_err is not None:
        pivot_tables.append(pivot_err)

    results_table = pd.concat(pivot_tables, axis=1)

    # Compute overall weighted averages using class weights directly on raw rows
    if include_overall:
        metrics = []
        # Pass@1: weighted average over df_pass_at_1_for_nmax rows
        if metrics_to_include is None or pass_at_1_abbr in metrics_to_include:
            overall_pass = df_pass_at_1_for_nmax.groupby(
                "method",
                observed=False,
            ).apply(
                lambda g: np.average(
                    g["pass@k"], weights=g["task_type"].map(class_weights)
                ),
                include_groups=False,
            )
            metrics.append(pass_at_1_abbr)

        # Coherency: weighted average over df_samples rows
        if pivot_coh is not None:
            # Coherency: weighted average over df_samples rows, ignore NaN scores
            overall_coh = df_samples.groupby("method", observed=False).apply(
                lambda g: np.average(
                    # filter out NaNs
                    g.loc[g["coherency_score"].notna(), "coherency_score"],
                    weights=g.loc[g["coherency_score"].notna(), "task_type"].map(
                        class_weights
                    ),
                ),
                include_groups=False,
            )
            metrics.append(coherency_abbr)

        # Error: weighted average over df_pass_at_1_for_nmax rows
        if include_error:
            overall_err = df_pass_at_1_for_nmax.groupby(
                "method",
                observed=False,
            ).apply(
                lambda g: np.average(
                    g["error_true"].astype(float),
                    weights=g["task_type"].map(class_weights),
                ),
                include_groups=False,
            )
            metrics.append(error_abbr)

        # Build overall DataFrame
        overall_columns = []
        overall_data = []

        if metrics_to_include is None or pass_at_1_abbr in metrics_to_include:
            overall_columns.append(("Overall", pass_at_1_abbr))
            overall_data.append(overall_pass)

        if pivot_coh is not None:
            overall_columns.append(("Overall", coherency_abbr))
            overall_data.append(overall_coh)

        if include_error:
            overall_columns.append(("Overall", error_abbr))
            overall_data.append(overall_err)

        overall_df = pd.concat(overall_data, axis=1)
        overall_df.columns = pd.MultiIndex.from_tuples(overall_columns)

        # Prepend overall columns to result table
        results_table = pd.concat([overall_df, results_table], axis=1)

    # Reorder columns: for each task type (including 'Overall' if included)
    all_task_types = sorted(
        results_table.columns.levels[0].drop("Overall")
        if include_overall
        else results_table.columns.levels[0]
    )
    if include_overall:
        all_task_types.append("Overall")

    ordered_columns = [(task, metric) for task in all_task_types for metric in metrics]
    results_table = results_table.reindex(columns=ordered_columns)

    # Combine metrics in one column if requested
    if combine_metrics and len(metrics) > 1:
        # Create a new results table with combined metrics
        new_columns = []
        new_data = {}

        for task in all_task_types:
            new_col = f"{task}"
            new_columns.append(new_col)

            for method in results_table.index:
                if method not in new_data:
                    new_data[method] = {}

                metric_values = []
                for metric in metrics:
                    if (task, metric) in results_table.columns:
                        val = results_table.loc[method, (task, metric)]
                        metric_values.append(f"{val:.{round_digits}f}")

                new_data[method][new_col] = " / ".join(metric_values)

        # Create the new DataFrame
        combined_table = pd.DataFrame(new_data).T
        combined_table.columns = new_columns
        results_table = combined_table

    # Round the values
    if not combine_metrics:
        results_table = results_table.round(round_digits)

    results_table.index.name = None

    # Add method information if requested
    if include_method_info:
        # Create a list of new indices with the detailed information and gather sort order
        new_indices = []
        sort_orders = []
        for method in results_table.index:
            # Find the pipeline_name that corresponds to this method
            # Since there's now a one-to-many relationship, we need to search through the dataframes
            matching_rows = df_pass_at_1_for_nmax[
                df_pass_at_1_for_nmax["method"] == method
            ]
            if not matching_rows.empty:
                pipeline_name = matching_rows["pipeline_name"].iloc[0]
                # Get the hf_model from the data
                hf_model = matching_rows.get("hf_model", pd.Series([None])).iloc[0]
                if hf_model is None:
                    hf_model = "Llama-3.2-1B"  # default

                try:
                    info = get_pipeline_info(pipeline_name, hf_model)
                    new_index = (
                        info["name"],
                        info["sampling_method"],
                        info["model_short"],
                    )
                    # Use the pipeline order from ALL_PIPELINES
                    sort_orders.append(
                        ALL_PIPELINES.index(pipeline_name)
                        if pipeline_name in ALL_PIPELINES
                        else 999
                    )
                except KeyError:
                    new_index = (method, "", "")
                    sort_orders.append(999)
            else:
                new_index = (method, "", "")
                sort_orders.append(999)
            new_indices.append(new_index)

        # Replace the index with a MultiIndex
        results_table.index = pd.MultiIndex.from_tuples(
            new_indices, names=["Method", "Sampling Method", "Model"]
        )
        # Reorder the rows based on the ordering in PIPELINE_INFO
        results_table["__order__"] = sort_orders
        results_table = results_table.sort_values(by="__order__")
        results_table = results_table.drop(columns="__order__")

    # Transpose the table if requested
    if transpose_table:
        results_table = results_table.T

    return results_table
