"""
arc_thesis_plots.py

Parses the two kinds of logs your pipeline produces and turns them into
plots you can drop straight into the thesis.

  1. EVALUATION logs  (printed by selection.EvalTool, i.e. anything that
     runs run_evaluation_Nemo-full.py), per-task lines like:

         * task '00576224_0': 8 candidates, correct solution FOUND
           max_gen_prob:  100.00% (  1.00/  1.00), corr_sol. @ 1 / 8
           max_aug_prob:  100.00% (  1.00/  1.00), corr_sol. @ 1 / 8
           ...
           correct_found:  60.00% (  3.00/  5.00)

     -> parse_eval_log() extracts, per task, whether each of the 6 scoring
        algorithms solved it (rank <= n_guesses) and whether the correct
        solution was in the candidate pool at all (the generation ceiling).

  2. TRAINING DYNAMICS logs (printed by HF/Unsloth's GRPOTrainer during
     run_evaluation_Nemo-full.py's GRPO stage), one python-dict-literal
     per logging step:

         {'loss': 0.0132, 'grad_norm': ..., 'reward': -0.9, 'reward_std': 0.0,
          'frac_reward_zero_std': 1.0, 'kl': 0.33, 'completions/mean_length': 91.0,
          'rewards/extrinsic_reward_fn/mean': -0.9, ...}

     -> parse_training_log() reads every such line into one wide DataFrame,
        automatically picking up whatever reward-function columns exist
        (so it keeps working once you add confidence_reward_fn /
        curiosity_reward_fn columns back in).

Usage
-----
    from arc_thesis_plots import *

    eval_df   = parse_eval_log("eval_output_0_25.log")
    train_df  = parse_training_log("eval_output_0_25_temp06_ms700_k4_r3.log")

    plot_eval_summary(eval_df, title="Stage-1 only (no GRPO)", save_path="stage1_summary.png")
    plot_eval_convergence(eval_df, title="Stage-1 only", save_path="stage1_convergence.png")
    plot_training_dynamics(train_df, save_path="grpo_dynamics.png")

    plot_condition_comparison({
        "Stage-1 only":        summarize_eval(parse_eval_log("eval_output_0_25.log")),
        "GRPO extrinsic-only": summarize_eval(parse_eval_log("other_eval.log")),
    }, save_path="condition_comparison.png")
"""

import ast
import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SELECTION_ALGOS = [
    'max_gen_prob', 'max_aug_prob', 'min_aug_prob',
    'sum_aug_prob', 'mul_aug_prob', 'mul_all_prob',
]

_TASK_RE = re.compile(
    r"^\s*\*\s*task '([^']+)':\s*(\d+) candidates, correct solution (FOUND|not found)"
)
_ALGO_RE = re.compile(
    r"^\s*([\w/]+):\s*[\d.]+%\s*\(\s*[\d.]+\s*/\s*[\d.]+\)(?:, corr_sol\. @\s*(\d+)\s*/\s*(\d+))?"
)


# --------------------------------------------------------------------------
# 1. Evaluation logs (candidate-selection accuracy)
# --------------------------------------------------------------------------

def parse_eval_log(path, n_guesses=2):
    """Returns one row per unique task (last occurrence wins, so log
    restarts / resilient_inference retries don't create duplicates)."""
    results = OrderedDict()
    order = []
    current_task = None

    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _TASK_RE.match(line)
            if m:
                name, n_cand, found = m.group(1), int(m.group(2)), m.group(3)
                current_task = name
                if name not in results:
                    order.append(name)
                results[name] = {"candidates": n_cand, "correct_found": found == "FOUND"}
                continue
            if current_task is not None:
                am = _ALGO_RE.match(line)
                if am:
                    algo, rank = am.group(1), am.group(2)
                    if algo in SELECTION_ALGOS:
                        results[current_task][algo] = int(rank) if rank is not None else None

    rows = []
    for i, name in enumerate(order):
        r = results[name]
        row = {
            "task": name,
            "order": i,
            "candidates": r.get("candidates"),
            "correct_found": r.get("correct_found", False),
        }
        for a in SELECTION_ALGOS:
            rank = r.get(a)
            row[a] = bool(rank is not None and rank <= n_guesses)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No EvalTool task blocks found in {path}. Wrong log type?")
    return df


def summarize_eval(df):
    """Final accuracy (%) per selection algorithm, plus the correct_found
    ceiling (how often the right answer was *generated at all*, regardless
    of selection). Sorted best-to-worst."""
    s = {"correct_found_ceiling": df["correct_found"].mean() * 100}
    for a in SELECTION_ALGOS:
        s[a] = df[a].mean() * 100
    return pd.Series(s).sort_values(ascending=False)


def plot_eval_summary(df, title="", save_path=None, n_guesses=2):
    s = summarize_eval(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#888888" if k == "correct_found_ceiling" else "#4C72B0" for k in s.index]
    ax.bar(s.index, s.values, color=colors)
    for i, v in enumerate(s.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel(f"% of {len(df)} tasks solved (top-{n_guesses} guesses)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_eval_convergence(df, title="", save_path=None):
    """Cumulative accuracy per algorithm as more tasks are processed,
    useful to show stability (paper's Table 6 style comparison over time)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df["order"] + 1
    for a in SELECTION_ALGOS:
        ax.plot(x, df[a].expanding().mean() * 100, label=a, linewidth=1.3)
    ax.plot(x, df["correct_found"].expanding().mean() * 100,
            label="correct_found (ceiling)", color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("# tasks processed")
    ax.set_ylabel("cumulative accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_condition_comparison(summaries: dict, save_path=None, title="Selection accuracy by condition"):
    """summaries: {"condition name": summarize_eval(df), ...}"""
    algos = list(next(iter(summaries.values())).index)
    x = np.arange(len(algos))
    width = 0.8 / max(len(summaries), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, s) in enumerate(summaries.items()):
        vals = [s.get(a, np.nan) for a in algos]
        ax.bar(x + i * width, vals, width=width, label=name)
    ax.set_xticks(x + width * (len(summaries) - 1) / 2)
    ax.set_xticklabels(algos, rotation=30, ha="right")
    ax.set_ylabel("% of tasks solved")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# 2. GRPO training-dynamics logs
# --------------------------------------------------------------------------

def parse_training_log(path):
    """Reads every printed trainer-log dict line into one wide DataFrame.
    Works for BOTH log flavors your pipeline produces:
      - plain SFT logs from run_finetuning_Nemo-full.py (Unsloth/HF Trainer):
            {'loss': 0.41, 'grad_norm': 1.2, 'learning_rate': 8e-5, 'epoch': 0.3}
      - GRPO logs from run_evaluation_Nemo-full.py's GRPOTrainer:
            {'loss': 0.0132, 'reward': -0.9, 'kl': 0.33, 'frac_reward_zero_std': 1.0, ...}
    Column set adapts automatically to whichever fields exist, so the same
    function/plots work for either training stage."""
    rows = []
    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("{'loss'"):
                continue
            try:
                d = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if isinstance(d, dict) and "loss" in d:
                rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No trainer-log dict lines found in {path}. Wrong log type?")
    df["log_index"] = range(len(df))
    return df


def plot_training_dynamics(df, columns=None, save_path=None, title=None, smooth_window=None, x_col=None):
    """smooth_window: if set (e.g. 25), overlays a rolling mean on top of
    the raw (faint) series. Per-step logs are extremely spiky, so a
    smoothed line is usually what you actually want in the thesis figure.

    x_col: force the x-axis column ("epoch" or "log_index"). If your log
    concatenates multiple shard runs, 'epoch' resets at each shard boundary,
    pass x_col="log_index" to get a single monotonic x-axis instead.

    columns: which fields to plot. If omitted, auto-picks whichever of
    ["loss", "learning_rate", "reward", "frac_reward_zero_std", "kl",
    "completions/mean_length", "grad_norm"] are actually present, so the
    same call works for a plain SFT finetuning log (loss/learning_rate/
    grad_norm only) or a GRPO log (adds reward/kl/frac_reward_zero_std)."""
    if columns is None:
        candidates = ["loss", "learning_rate", "reward", "frac_reward_zero_std",
                      "kl", "completions/mean_length", "grad_norm"]
        columns = [c for c in candidates if c in df.columns]
    if x_col is None:
        x_col = "epoch" if "epoch" in df.columns else "log_index"
    x = df[x_col]
    xlabel = x_col

    fig, axes = plt.subplots(len(columns), 1, figsize=(9, 2.3 * len(columns)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, columns):
        if smooth_window:
            ax.plot(x, df[col], linewidth=0.5, alpha=0.3, color="#4C72B0")
            ax.plot(x, df[col].rolling(smooth_window, min_periods=1).mean(),
                     linewidth=1.5, color="#C44E52")
        else:
            ax.plot(x, df[col], linewidth=0.8)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(xlabel)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_reward_components(df, save_path=None, title=None):
    """Plots the mean of every rewards/<fn_name>/mean column found,
    i.e. extrinsic vs confidence vs curiosity contribution over training."""
    reward_cols = [c for c in df.columns if c.startswith("rewards/") and c.endswith("/mean")]
    if not reward_cols:
        raise ValueError("No per-reward-function columns found (rewards/<name>/mean).")
    x = df["epoch"] if "epoch" in df.columns else df["log_index"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in reward_cols:
        label = col.split("/")[1]
        ax.plot(x, df[col], label=label, linewidth=1.0)
    ax.set_xlabel("epoch" if "epoch" in df.columns else "logged step")
    ax.set_ylabel("reward component mean")
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


# --------------------------------------------------------------------------
# 3. High-level entry points -- these are the ones you'll normally call
# --------------------------------------------------------------------------

def analyze_run(path, label=None, out_dir=None, n_guesses=2, smooth_window=25, verbose=True):
    """Point this at ONE log file. It auto-detects what's inside (EvalTool
    blocks, GRPO training-dynamics lines, or both, a single log can
    contain both, e.g. a GRPO run that trains and then evaluates) and
    produces whichever plots apply.

    Returns a dict: {"label", "path", "eval_df", "eval_summary", "train_df"}
    (eval_df/eval_summary/train_df are only present if that content was found).
    """
    label = label or os.path.basename(path)
    result = {"label": label, "path": path}

    try:
        eval_df = parse_eval_log(path, n_guesses=n_guesses)
        result["eval_df"] = eval_df
        result["eval_summary"] = summarize_eval(eval_df)
        if verbose:
            print(f"[{label}] evaluation data: {len(eval_df)} tasks")
            print(result["eval_summary"].round(1).to_string(), "\n")
    except ValueError:
        pass

    try:
        train_df = parse_training_log(path)
        result["train_df"] = train_df
        if verbose:
            print(f"[{label}] training-dynamics data: {len(train_df)} logged steps\n")
    except ValueError:
        pass

    if "eval_df" not in result and "train_df" not in result:
        raise ValueError(
            f"'{path}' doesn't look like either log type, no EvalTool task "
            f"blocks and no GRPO trainer-log dict lines found."
        )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        safe = re.sub(r"[^\w.-]", "_", label)

        if "eval_df" in result:
            plot_eval_summary(
                result["eval_df"], title=f"{label} - selection accuracy",
                save_path=os.path.join(out_dir, f"{safe}_eval_summary.png"), n_guesses=n_guesses,
            )
            plot_eval_convergence(
                result["eval_df"], title=f"{label} - cumulative accuracy",
                save_path=os.path.join(out_dir, f"{safe}_eval_convergence.png"),
            )

        if "train_df" in result:
            plot_training_dynamics(
                result["train_df"], title=f"{label} - training dynamics",
                save_path=os.path.join(out_dir, f"{safe}_training_dynamics.png"),
                smooth_window=smooth_window, x_col="log_index",
            )
            reward_cols = [c for c in result["train_df"].columns
                           if c.startswith("rewards/") and c.endswith("/mean")]
            if len(reward_cols) > 1:  # only meaningful once you have >1 reward function active
                plot_reward_components(
                    result["train_df"], title=f"{label} - reward components",
                    save_path=os.path.join(out_dir, f"{safe}_reward_components.png"),
                )

    return result


def compare_runs(logs, out_dir=None, n_guesses=2, smooth_window=25, verbose=True):
    """Point this at SEVERAL logs to compare conditions (e.g. Stage-1-only
    vs. GRPO extrinsic-only vs. GRPO extrinsic+intrinsic).

    `logs` can be:
      - a dict of {label: path}, e.g.
            {"Stage-1 only": "eval_output_0_25.log",
             "GRPO extrinsic-only": "eval_output_0_25_temp06_ms700_k4_r3.log"}
      - or a plain list of paths (the filename is used as the label).

    Runs analyze_run() on each log individually (same per-run plots as
    calling it directly), and additionally produces one combined bar chart
    (condition_comparison.png) across every log that had evaluation data.

    Returns {label: result_dict_from_analyze_run}.
    """
    if isinstance(logs, (list, tuple)):
        logs = {os.path.basename(p): p for p in logs}

    results = {}
    for label, path in logs.items():
        results[label] = analyze_run(
            path, label=label, out_dir=out_dir,
            n_guesses=n_guesses, smooth_window=smooth_window, verbose=verbose,
        )

    eval_summaries = {label: r["eval_summary"] for label, r in results.items() if "eval_summary" in r}
    if len(eval_summaries) >= 2:
        save_path = os.path.join(out_dir, "condition_comparison.png") if out_dir else None
        plot_condition_comparison(
            eval_summaries, save_path=save_path,
            title="Selection accuracy by condition",
        )
    elif verbose and len(eval_summaries) == 1:
        print("Only one run had evaluation data, skipping the comparison chart "
              "(need >= 2 runs with eval data to compare).")

    return results
