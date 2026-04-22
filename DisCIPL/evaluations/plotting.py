import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from evaluations.analysis_utils import get_pipeline_info

PIPELINE_PALETTE = {
    "disciple_smc": "tab:green",
    "disciple_oracle_smc": "tab:green",
    "disciple_importance": "tab:orange",
    "disciple_oracle_importance": "tab:orange",
    "disciple_rejection": "tab:red",
    "disciple_oracle_rejection": "tab:red",
    "vllm": "tab:pink",
    "vllm_beam_search": "tab:purple",
    "huggingface": "tab:pink",
    "huggingface_beam_search": "tab:purple",
    "gpt-4o-mini": "tab:cyan",
    "gpt-4o": "tab:blue",
    "gpt-4o-cot": "tab:blue",
    "o1": "tab:gray",
}


def build_method_palette(df, palette=PIPELINE_PALETTE):
    method_palette = {}
    for pipeline_name, color in palette.items():
        # Get all methods associated with this pipeline
        methods = df[df["pipeline_name"] == pipeline_name]["method"].unique()
        for method in methods:
            method_palette[method] = color
    return method_palette


def plot_pass_at_k(
    df_pass_at_k,
    dataset,
    subtasks,
    x_label="Sampling Budget (N)",
    y_label="Pass@k",
    palette=None,
    dash_methods=[],
    fig_height=6,
    fig_width=12,
    height_ratios=[2, 1],
    use_class_weights=True,
    show_group_labels=True,
):

    N_max = df_pass_at_k["N"].max()

    df_pass_at_nmax = df_pass_at_k[(df_pass_at_k["N"] == N_max)]

    # Identify unique task types for creating individual plots
    task_types = df_pass_at_k["task_type"].unique()
    n_tasks = len(task_types)

    # Define the figure size
    fig = plt.figure(figsize=(fig_width, fig_height))

    # -------------------- Add Supertitle -------------------- #
    # fig.suptitle(f"{dataset} {subtasks}", fontsize=18)

    # Create an outer GridSpec with 2 rows:
    # Top row for bar plot and overall line plot; bottom row for task-level plots.
    gs_outer = gridspec.GridSpec(
        nrows=2, ncols=1, height_ratios=height_ratios, figure=fig, hspace=0.3
    )

    # -------------------- Top Row: Two Columns -------------------- #
    # Split the top row into 2 columns
    gs_top = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=gs_outer[0], wspace=0.2
    )

    # Left subplot: bar plot
    ax_bar = fig.add_subplot(gs_top[0])
    sns.barplot(
        y="pass@k",
        weights="class_weight" if use_class_weights else None,
        data=df_pass_at_nmax,
        hue="method",
        palette=palette,
        ax=ax_bar,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
    )
    # Add cross-hatch pattern to bars for methods in dash_methods
    # Note: The patches are in order corresponding to unique hue levels.
    # Retrieve the order of methods as they appear in the plot.
    methods = df_pass_at_nmax["method"].unique()
    for patch, method in zip(ax_bar.patches, methods):
        if method in dash_methods:
            patch.set_hatch("///")

    ax_bar.set_title(f"{y_label} for N={N_max}", fontsize=14)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel(y_label, fontsize=14)
    ax_bar.tick_params(axis="both", which="major", labelsize=12, length=6)
    sns.despine(ax=ax_bar, top=True, right=True)
    ax_bar.get_legend().remove()
    ax_bar.set_ylim(0, 1)  # Ensure y-axis is [0, 1]

    import matplotlib.path as mpath
    from matplotlib.patches import PathPatch

    if show_group_labels:
        # Get the left and right boundaries of each bar (data coords)
        bar_lefts = [patch.get_x() for patch in ax_bar.patches]
        bar_rights = [patch.get_x() + patch.get_width() for patch in ax_bar.patches]

        # Define your groups by slicing indices:
        # First 6 bars: "DisCIPL", next 2: "Follower", next 2: "Planner"
        groups = [
            (0, 6, "DisCIPL"),
            (6, 8, "Follower"),
            (8, 10, "Planner"),
            (10, 11, "o1"),
        ]

        def curly_brace(x0, x1, y, height, flip=True, sharpness=0.75):
            """
            Creates a curly brace with two cubic Bezier segments that form a pointy, sigmoidal shape.
            """
            Path = mpath.Path
            sign = -1 if flip else 1
            mid = (x0 + x1) / 2

            # First cubic segment from left to mid.
            p0 = (x0, y)
            p1 = (x0, y + sign * height)
            p2 = (mid, sharpness * y)
            p3 = (mid, y + sign * height)

            # Second cubic segment from mid to right.
            p4 = (mid, sharpness * y)
            p5 = (x1, y + sign * height)
            p6 = (x1, y)

            verts = [p0, p1, p2, p3, p4, p5, p6]
            codes = [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
            ]
            return mpath.Path(verts, codes)

        for start, end, label in groups:
            # Get the left edge of the first bar and right edge of the last bar in the group
            x0 = bar_lefts[start]
            x1 = bar_rights[end - 1]
            group_center = (x0 + x1) / 2
            dx = 0.01

            # Draw an upward-facing curly brace (flip=True) as a PathPatch.
            brace_path = curly_brace(x0 + dx, x1 - dx, y=-0.01, height=0.025, flip=True)
            brace_patch = PathPatch(
                brace_path,
                transform=ax_bar.get_xaxis_transform(),
                linewidth=1.0,
                edgecolor="black",
                fill=False,  # use fill=False so only the line is drawn
                clip_on=False,  # disable clipping so it's visible outside the axis bounds
            )
            ax_bar.add_patch(brace_patch)

            # Place the group label below the brace.
            ax_bar.text(
                group_center,
                -0.05,  # offset from the bottom of the plot
                label,
                ha="center",
                va="top",
                transform=ax_bar.get_xaxis_transform(),
                fontsize=10,
                clip_on=False,  # disable text clipping
            )

    # Right subplot: Existing Overall Line Plot
    ax_overall = fig.add_subplot(gs_top[1])
    sns.lineplot(
        data=df_pass_at_k,
        x="N",
        y="pass@k",
        hue="method",
        palette=palette,
        ax=ax_overall,
        errorbar=None,
        weights="class_weight" if use_class_weights else None,
    )

    def _set_dashed_line_styles(ax, dash_methods):
        lines = ax.get_lines()
        assert len(lines) % 2 == 0
        plot_lines = lines[: len(lines) // 2]
        legend_lines = lines[len(lines) // 2 :]
        for plot_line, legend_line in zip(plot_lines, legend_lines):
            label = legend_line.get_label()
            if label in dash_methods:
                plot_line.set_linestyle("--")
                legend_line.set_linestyle("--")

    _set_dashed_line_styles(ax_overall, dash_methods)

    ax_overall.set_xticks(df_pass_at_k["N"].unique())
    ax_overall.set_xticklabels(df_pass_at_k["N"].unique())
    ax_overall.set_title(f"{y_label} for varying N", fontsize=14)
    plt.xlabel("")
    # ax_overall.set_xlabel(x_label, fontsize=12, labelpad=0)
    ax_overall.set_ylabel(y_label, fontsize=14)
    ax_overall.tick_params(axis="both", which="major", labelsize=12, length=6)
    sns.despine(ax=ax_overall, top=True, right=True)
    ax_overall.get_legend().remove()
    ax_overall.set_ylim(0, 1)  # Ensure y-axis is [0, 1]

    # -------------------- Bottom Row: Task-Level Plots -------------------- #
    gs_tasks = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=n_tasks, subplot_spec=gs_outer[1], wspace=0.15
    )
    axes_tasks = []

    for i, task in enumerate(task_types):
        ax = fig.add_subplot(gs_tasks[i])
        subset = df_pass_at_k[df_pass_at_k["task_type"] == task]
        sns.lineplot(
            data=subset,
            x="N",
            y="pass@k",
            hue="method",
            palette=palette,
            ax=ax,
            errorbar=None,
        )
        _set_dashed_line_styles(ax, dash_methods)

        ax.set_xticks(subset["N"].unique())
        ax.set_xticklabels(subset["N"].unique())
        ax.set_title(task, fontsize=12)
        ax.set_xlabel("")
        ax.set_yticks(ax_overall.get_yticks())
        if i == 0:
            ax.set_ylabel(y_label, fontsize=12)
        else:
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.set_yticklabels([])
        ax.tick_params(axis="both", which="major", labelsize=9, length=4)
        sns.despine(ax=ax, top=True, right=True)
        ax.get_legend().remove()

        axes_tasks.append(ax)

    # -------------------- Shared Legend -------------------- #
    handles, labels = ax_bar.get_legend_handles_labels()
    # Create custom handles to preserve hatch patterns.
    custom_handles = []
    for handle, label in zip(handles, labels):
        hatch_value = "///" if label in dash_methods else ""
        custom_handles.append(
            Patch(
                facecolor=handle.get_facecolor(),
                hatch=hatch_value,
                edgecolor=handle.get_edgecolor(),
                label=label,
            )
        )
    fig.legend(
        custom_handles,
        labels,
        loc="lower center",
        ncol=(len(labels) + 1) // 2,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=12,
    )

    plt.subplots_adjust(bottom=0.20, top=1.0)
    return fig


def plot_breakdown(df, x="task_type", col="method"):
    palette = {
        "Valid": "tab:green",
        "Invalid": "tab:red",
        "Error": "tab:grey",
    }

    g = sns.displot(
        data=df,
        x=x,
        hue="valid_breakdown",
        col=col,
        stat="proportion",
        discrete=True,
        multiple="fill",
        palette=palette,
        shrink=0.9,
        hue_order=list(palette.keys()),
    )

    for ax in g.axes.flat:
        ax.set_title(ax.get_title().split(" = ")[-1])

    return g


def plot_confusion_matrices(df, round_digits=2):

    # Filter the dataframe for the disciple pipelines
    df = df[df["pipeline_name"].str.contains("disciple")]

    # Get the ordered categories from the method column
    categories = df["method"].unique().tolist()

    # Create subplots: arrange them in a row with shared y-axis
    fig, axs = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4))

    # In case there's only one category, wrap axs in a list
    if len(categories) == 1:
        axs = [axs]

    def map_pred(val):
        if pd.isnull(val):
            return "Null"
        return "Valid" if val else "Invalid"

    for i, (ax, method) in enumerate(zip(axs, categories)):
        # Filter the dataframe for the current method
        df_method = df[df["method"] == method]

        # Map the ground-truth and predictions
        y_true = df_method["valid"].apply(map_pred)
        y_pred = df_method["check_result"].apply(map_pred)

        # Compute the confusion matrix with an extra label for NaN predictions.
        labels = ["Valid", "Invalid", "Null"]
        cm = confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            normalize="all",
        )

        if round_digits is not None:
            cm = np.round(cm, round_digits)

        # Display the confusion matrix on the current axis
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False)

        # Remove the ticks and adjust label alignment (we will override them next)
        ax.tick_params(axis="both", which="both", length=0, labelsize=34)

        # Replace xticklabels and yticklabels with custom symbols
        custom_labels = ["✔", "✖", "∅"]
        label_colors = ["green", "red", "gray"]

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(custom_labels)
        for tick, color in zip(ax.get_xticklabels(), label_colors):
            tick.set_color(color)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(custom_labels, va="center")
        for tick, color in zip(ax.get_yticklabels(), label_colors):
            tick.set_color(color)

        ax.set_xlabel("Predicted", fontsize=28)

        if i == 0:
            ax.set_ylabel("True", fontsize=28)
        else:
            ax.set_ylabel("")

        # Set a title for the plot
        ax.set_title(method, fontsize=28)

    plt.tight_layout()
    return fig


def plot_error_distribution(df_all_attempts, pipeline_name, top_n=20):
    top_n = 20

    df_pipeline = df_all_attempts[
        df_all_attempts["pipeline_name"] == pipeline_name
    ].copy()
    df_pipeline.loc[:, "error"] = df_pipeline.loc[:, "error"].fillna("No error")

    # Get the top N most frequent errors
    n_unique_errors = df_pipeline["error"].nunique()
    top_errors = (
        df_pipeline["error"]
        .value_counts()
        .nlargest(min(top_n - 1, n_unique_errors))
        .index
    )

    # Map all other errors to "Other"
    df_pipeline.loc[~df_pipeline["error"].isin(top_errors), "error"] = "Other"

    # Order the hue by error frequency
    error_order = df_pipeline["error"].value_counts().index

    with sns.plotting_context("talk", font_scale=0.75):
        # Define a custom palette based on tab20 with green as the first color.
        palette_custom = ["tab:green"] + sns.color_palette("tab20")[
            6 : 6 + len(error_order) - 1
        ]
        g = sns.displot(
            data=df_pipeline,
            x="task_type",
            hue="error",
            hue_order=error_order,
            stat="proportion",
            discrete=True,
            multiple="fill",
            shrink=0.8,
            height=6,
            palette=palette_custom,
        )

    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.ylabel("Error Rate")

    legend = g.legend
    legend.set_title("")

    return g


def plot_coherency(
    df_samples, show_y_axis=False, rescale_y_axis=False, kind="kde", legend=True
):

    # Determine the unique rows and columns
    task_types = sorted(df_samples["task_type"].unique())
    methods = list(df_samples["method"].unique())
    nrows, ncols = len(task_types), len(methods)

    with sns.plotting_context("talk", font_scale=0.75):

        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=(ncols * 3.0, nrows * 1.0))
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.1, hspace=0.3)

        ymax = 0
        for i, task in enumerate(task_types):
            for j, method in enumerate(methods):
                ax = fig.add_subplot(gs[i, j])
                subset = df_samples[
                    (df_samples["task_type"] == task) & (df_samples["method"] == method)
                ]

                if kind == "kde":
                    # Plot KDE for each valid_true group with different colors
                    sns.kdeplot(
                        data=subset,
                        x="coherency_score",
                        fill=True,
                        common_norm=True,
                        clip=(0, 10),
                        hue="valid_true",
                        palette={True: "tab:green", False: "tab:red"},
                        ax=ax,
                        legend=None,
                    )
                elif kind == "hist":
                    # Plot histograms for each valid_true group with different colors
                    sns.histplot(
                        data=subset,
                        x="coherency_score",
                        bins=10,
                        binrange=(0, 10),
                        common_norm=True,
                        hue="valid_true",
                        palette={True: "tab:green", False: "tab:red"},
                        ax=ax,
                        kde=True,
                        legend=None,
                    )
                else:
                    raise ValueError(f"Invalid kind: {kind}")

                ax.set_xlim(0, 10)
                ymax = max(ymax, ax.get_ylim()[1])

                # Despine top and right
                sns.despine(ax=ax, top=True, left=(not show_y_axis), right=True)

                # Add titles/labels to the top row and left column only
                if i == 0:
                    ax.set_title(method)
                if j == 0:
                    ax.set_ylabel(task)
                else:
                    ax.set_ylabel("")

                # Remove x-axis labels for all but the bottom row
                if i == nrows - 1:
                    ax.set_xlabel("Coherency")
                    ax.set_xticks([0, 2, 4, 6, 8, 10])
                    ax.set_xticklabels(labels=[0, 2, 4, 6, 8, 10], fontsize=12)
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])

                if not show_y_axis:
                    ax.set_yticks([])
                    ax.set_yticklabels([])

        if rescale_y_axis:
            for i, task in enumerate(task_types):
                for j, method in enumerate(methods):
                    ax = fig.axes[i * ncols + j]
                    ax.set_ylim(0, ymax)

        if legend:
            unique = {
                True: Patch(
                    facecolor=to_rgba("tab:green", 0.25),
                    edgecolor="tab:green",
                    label="True",
                ),
                False: Patch(
                    facecolor=to_rgba("tab:red", 0.25),
                    edgecolor="tab:red",
                    label="False",
                ),
            }
            fig.legend(
                list(unique.values()),
                list(unique.keys()),
                loc="center right",
                bbox_to_anchor=(0.96, 0.5),
                title="Valid",
            )

        return fig
