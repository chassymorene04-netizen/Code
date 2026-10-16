from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, OUTPUTS_DIR


EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
TABLES_DIR = OUTPUTS_DIR / "tables"

SCENARIO_ORDER = ["risk_exaggeration", "selective_interpretation", "emotional_wording"]
EXP1_SCENARIO_ORDER = [*SCENARIO_ORDER, "compound_amplification"]
SCENARIO_LABELS = {
    "risk_exaggeration": "风险夸大",
    "selective_interpretation": "选择性解释",
    "emotional_wording": "情绪化措辞",
    "compound_amplification": "复合强化",
}
SCENARIO_LABELS_EN = {
    "risk_exaggeration": "Risk exaggeration",
    "selective_interpretation": "Selective interpretation",
    "emotional_wording": "Emotional wording",
    "compound_amplification": "Composite amplification",
}
LEVEL_ORDER = ["low", "medium", "high"]
LEVEL_LABELS = {"low": "低强度", "medium": "中强度", "high": "高强度"}
PLATFORM_ORDER = [
    "baseline_platform",
    "low_platform_amplification",
    "medium_platform_amplification",
    "high_platform_amplification",
]
EMOTION_RESPONSE_ORDER = ["anxiety", "anger", "suspicion", "none"]
EMOTION_RESPONSE_LABELS = {
    "anxiety": "焦虑/恐惧",
    "anger": "愤怒/厌恶",
    "suspicion": "怀疑/不安",
    "none": "无明显负面情绪",
}
EMOTION_RESPONSE_LABELS_EN = {
    "anxiety": "Anxiety",
    "anger": "Anger",
    "suspicion": "Suspicion",
    "none": "No salient negative emotion",
}
PLATFORM_LABELS = {
    "baseline_platform": "基准平台",
    "low_platform_amplification": "轻度放大",
    "medium_platform_amplification": "中等平台放大",
    "high_platform_amplification": "高平台放大",
}
PLATFORM_LABELS_EN = {
    "baseline_platform": "Baseline",
    "low_platform_amplification": "Low",
    "medium_platform_amplification": "Moderate",
    "high_platform_amplification": "High",
}


def _setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )


def _save_manuscript_bundle(fig: plt.Figure, output_stem: Path, *, dpi: int = 300) -> dict[str, str]:
    """Save the one canonical manuscript figure in preview, print, and editable formats."""

    _ensure_dir(output_stem.parent)
    png_path = output_stem.with_suffix(".png")
    tiff_path = output_stem.with_suffix(".tiff")
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(
        tiff_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    return {
        "png": str(png_path),
        "tiff": str(tiff_path),
        "svg": str(svg_path),
    }


EXP2_LEVEL_COLORS = {"low": "#7FA8C9", "medium": "#E7B866", "high": "#C7685A"}
EXP1_SCENARIO_COLORS = {
    "risk_exaggeration": "#E9605B",
    "selective_interpretation": "#F2A12A",
    "emotional_wording": "#5487B8",
    "compound_amplification": "#7B6AA2",
}
EXP1_SCENARIO_STYLES = {
    "risk_exaggeration": {"linestyle": "-", "marker": "o"},
    "selective_interpretation": {"linestyle": "--", "marker": "s"},
    "emotional_wording": {"linestyle": "-.", "marker": "^"},
    "compound_amplification": {"linestyle": ":", "marker": "D"},
}
EMOTION_COMPOSITION_COLORS = {
    "anxiety": "#1F77B4",
    "anger": "#FF7F0E",
    "suspicion": "#2CA02C",
    "none": "#E31A1C",
}
EXP5_SCALE_COLORS = {
    200: "#7EAD83",
    300: "#E0A458",
    400: "#BF6B62",
    500: "#8D78B7",
}
EXP6_PROFILE_COLORS = ["#5B8DB8", "#D9854C", "#759E6A", "#9A7BB8", "#B85C75"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _group_rows(rows: Iterable[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda item: _float(item, "step"))
    return grouped


def _split_typed_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Separate raw repetitions from aggregate rows in the compact output tables."""

    if not any(row.get("row_type") for row in rows):
        return [], rows
    return (
        [row for row in rows if row.get("row_type") == "run"],
        [row for row in rows if row.get("row_type") == "summary"],
    )


def _series(rows: list[dict[str, str]], value_key: str) -> tuple[list[float], list[float]]:
    ordered = sorted(rows, key=lambda item: _float(item, "step"))
    return [_float(row, "step") for row in ordered], [_float(row, value_key) for row in ordered]


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _matches(row: dict[str, str], filters: dict[str, str]) -> bool:
    return all(str(row.get(key, "")) == str(value) for key, value in filters.items())


def _step_std(
    raw_rows: list[dict[str, str]],
    value_key: str,
    filters: dict[str, str],
) -> dict[int, float]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in raw_rows:
        if _matches(row, filters):
            buckets[int(_float(row, "step"))].append(_float(row, value_key))
    return {step: _sample_std(values) for step, values in buckets.items()}


def _plot_exp1_manuscript_figure(
    rows: list[dict[str, str]],
    figures_dir: Path,
) -> dict[str, str]:
    """Create manuscript Figure 3 as one canonical three-panel composite."""

    _setup_matplotlib()
    figure = plt.figure(figsize=(8.8, 6.6))
    grid = figure.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 0.43, 0.43],
        left=0.075,
        right=0.985,
        bottom=0.11,
        top=0.955,
        hspace=0.55,
        wspace=0.28,
    )
    trajectory_axes = [figure.add_subplot(grid[0, :2]), figure.add_subplot(grid[0, 2:])]
    composition_axes = [
        figure.add_subplot(grid[1, :2]),
        figure.add_subplot(grid[1, 2:]),
        figure.add_subplot(grid[2, :2]),
        figure.add_subplot(grid[2, 2:]),
    ]

    trajectory_specs = (
        (trajectory_axes[0], "mean_group_negative_emotion", "std_group_negative_emotion", "Group mean negative emotion, G(t)", "(a) Group mean negative emotion"),
        (trajectory_axes[1], "mean_high_negative_ratio", "std_high_negative_ratio", "Proportion of high-emotion agents", "(b) Proportion of high-emotion agents"),
    )
    for axis, mean_key, std_key, ylabel, title in trajectory_specs:
        for scenario in EXP1_SCENARIO_ORDER:
            scenario_rows = sorted(
                [row for row in rows if row.get("scenario_type") == scenario],
                key=lambda row: _float(row, "step"),
            )
            x = np.asarray([_float(row, "step") for row in scenario_rows], dtype=float)
            mean = np.asarray([_float(row, mean_key) for row in scenario_rows], dtype=float)
            std = np.asarray([_float(row, std_key) for row in scenario_rows], dtype=float)
            color = EXP1_SCENARIO_COLORS[scenario]
            style = EXP1_SCENARIO_STYLES[scenario]
            axis.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.18, linewidth=0)
            axis.plot(
                x,
                mean,
                color=color,
                linewidth=1.35,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=2,
                markersize=2.2,
                label=SCENARIO_LABELS_EN[scenario],
            )
        axis.set_title(title, loc="left", fontsize=9.2, fontweight="bold")
        axis.set_xlabel("Simulation step", fontsize=7.5)
        axis.set_ylabel(ylabel, fontsize=7.5)
        axis.set_xlim(0, 40)
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", color="#E4E9ED", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=6.8)
    trajectory_axes[0].legend(loc="upper left", fontsize=6.5, handlelength=2.3)

    emotion_keys = list(EMOTION_RESPONSE_ORDER)
    for axis, scenario in zip(composition_axes, EXP1_SCENARIO_ORDER):
        scenario_rows = sorted(
            [row for row in rows if row.get("scenario_type") == scenario],
            key=lambda row: _float(row, "step"),
        )
        x = np.asarray([_float(row, "step") for row in scenario_rows], dtype=float)
        for emotion in emotion_keys:
            mean = np.asarray([_float(row, f"mean_perceived_{emotion}_mean") for row in scenario_rows], dtype=float)
            std = np.asarray([_float(row, f"std_perceived_{emotion}_mean") for row in scenario_rows], dtype=float)
            color = EMOTION_COMPOSITION_COLORS[emotion]
            axis.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.12, linewidth=0)
            axis.plot(x, mean, color=color, linewidth=1.05, label=EMOTION_RESPONSE_LABELS_EN[emotion])
        axis.set_title(SCENARIO_LABELS_EN[scenario], fontsize=7.8, fontweight="bold", pad=2)
        axis.set_xlim(0, 40)
        axis.set_ylim(0, 1.0)
        axis.set_xlabel("Simulation step", fontsize=6.8)
        axis.set_ylabel("Mean proportion", fontsize=6.8)
        axis.grid(axis="y", color="#E4E9ED", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=6.1)
    figure.text(0.055, 0.555, "(c) Emotional composition", fontsize=9.2, fontweight="bold", ha="left")
    handles, labels = composition_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.54, 0.025), fontsize=6.6)

    bundle = _save_manuscript_bundle(figure, figures_dir / "figure_3_exp1_collective_emotion", dpi=300)
    plt.close(figure)
    return {
        "figure_3_collective_emotion": bundle["png"],
        "figure_3_collective_emotion_tiff": bundle["tiff"],
        "figure_3_collective_emotion_svg": bundle["svg"],
    }


def plot_experiment_1(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    _ensure_dir(figures_dir)
    time_series_path = Path(str(outputs["time_series_path"]))
    _, rows = _split_typed_rows(_read_csv(time_series_path))
    return _plot_exp1_manuscript_figure(rows, figures_dir)


def _plot_experiment_3_summary(
    rows: list[dict[str, str]],
    figures_dir: Path,
) -> dict[str, str]:
    """Create manuscript Figure 5 from the combined platform-strength experiment."""

    _ensure_dir(figures_dir)
    available_platforms = {row.get("platform_level", "") for row in rows}
    if not set(PLATFORM_ORDER).issubset(available_platforms):
        return {}
    _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.65))
    panel_specs = (
        ("mean_peak", "std_peak", "(a) Peak collective negative emotion"),
        ("mean_peak", "std_peak", "(b) Increase in peak relative to baseline"),
        ("mean_peak_average_rise_speed", "std_peak_average_rise_speed", "(c) Average rate of increase"),
    )
    x_values = np.arange(len(PLATFORM_ORDER), dtype=float)
    for panel_index, (axis, (metric, error_metric, title)) in enumerate(zip(axes, panel_specs)):
        for scenario in SCENARIO_ORDER:
            scenario_rows = {
                row.get("platform_level", ""): row
                for row in rows
                if row.get("scenario_type") == scenario
            }
            values = np.asarray([_float(scenario_rows[level], metric) for level in PLATFORM_ORDER], dtype=float)
            errors = np.asarray([_float(scenario_rows[level], error_metric) for level in PLATFORM_ORDER], dtype=float)
            if panel_index == 1:
                values = values - values[0]
            axis.errorbar(
                x_values,
                values,
                yerr=errors,
                color=EXP1_SCENARIO_COLORS[scenario],
                marker="o",
                markersize=3.8,
                linewidth=1.35,
                capsize=2.5,
                label=SCENARIO_LABELS_EN[scenario],
            )
        axis.set_title(title, loc="left", fontsize=8.6, fontweight="bold")
        axis.set_xticks(x_values)
        axis.set_xticklabels([PLATFORM_LABELS_EN[level] for level in PLATFORM_ORDER], rotation=24, ha="right")
        axis.grid(axis="y", color="#E4E9ED", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=6.8)
    axes[0].set_ylabel("Outcome value", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005), fontsize=7.0)
    fig.tight_layout(rect=[0.02, 0.12, 1, 1])
    bundle = _save_manuscript_bundle(fig, figures_dir / "figure_5_exp3_combined_platform_amplification", dpi=300)
    plt.close(fig)
    return {
        "figure_5_combined_platform_amplification": bundle["png"],
        "figure_5_combined_platform_amplification_tiff": bundle["tiff"],
        "figure_5_combined_platform_amplification_svg": bundle["svg"],
    }


def plot_experiment_3(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    _, rows = _split_typed_rows(_read_csv(str(outputs["summary_path"])))
    return _plot_experiment_3_summary(rows, figures_dir)


def _plot_figure6_manuscript(
    exp4_rows: list[dict[str, str]],
    exp5_summary_rows: list[dict[str, str]],
    exp5_time_rows: list[dict[str, str]],
    figures_dir: Path,
) -> dict[str, str]:
    """Create manuscript Figure 6 from sensitivity and population-size evidence."""

    if not exp4_rows or not exp5_summary_rows or not exp5_time_rows:
        return {}
    _setup_matplotlib()
    figure = plt.figure(figsize=(10.0, 6.65))
    outer = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.02, 0.88],
        left=0.055,
        right=0.985,
        bottom=0.13,
        top=0.95,
        hspace=0.28,
        wspace=0.30,
    )

    axis_a = figure.add_subplot(outer[0, 0])
    mechanism_rows = [row for row in exp4_rows if row.get("analysis_part") == "emotion_mechanism"]
    beta_values = sorted({_float(row, "beta_stimulus") for row in mechanism_rows})
    gamma_values = sorted({_float(row, "gamma_feedback") for row in mechanism_rows}, reverse=True)
    matrix = np.asarray(
        [
            [
                _float(next(row for row in mechanism_rows if np.isclose(_float(row, "beta_stimulus"), beta) and np.isclose(_float(row, "gamma_feedback"), gamma)), "mean_peak")
                for beta in beta_values
            ]
            for gamma in gamma_values
        ],
        dtype=float,
    )
    image = axis_a.imshow(matrix, cmap="Blues", aspect="auto", vmin=float(matrix.min()), vmax=float(matrix.max()))
    axis_a.set_xticks(np.arange(len(beta_values)), [f"{value:.2f}" for value in beta_values])
    axis_a.set_yticks(np.arange(len(gamma_values)), [f"{value:.2f}" for value in gamma_values])
    axis_a.set_xlabel("Information-stimulus weight", fontsize=7.2)
    axis_a.set_ylabel("Interaction-feedback weight", fontsize=7.2)
    axis_a.set_title("(a) Sensitivity of peak emotion to the\ninformation-stimulus and interaction-feedback weights", loc="left", fontsize=8.2, fontweight="bold")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis_a.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=6.4, color="white" if value > float(np.median(matrix)) else "#222222")
    colorbar = figure.colorbar(image, ax=axis_a, fraction=0.046, pad=0.04)
    colorbar.set_label("Peak collective negative emotion", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.2)

    axis_b = figure.add_subplot(outer[0, 1])
    decay_rows = sorted([row for row in exp4_rows if row.get("analysis_part") == "emotion_decay"], key=lambda row: _float(row, "emotion_decay"))
    memory_rows = sorted([row for row in exp4_rows if row.get("analysis_part") == "memory_weight"], key=lambda row: _float(row, "memory_emotion_weight"))
    decay_x = np.asarray([_float(row, "mean_peak") for row in decay_rows])
    decay_error = np.asarray([_float(row, "std_peak") for row in decay_rows])
    decay_y = np.asarray([_float(row, "emotion_decay") for row in decay_rows])
    memory_x = np.asarray([_float(row, "mean_peak") for row in memory_rows])
    memory_error = np.asarray([_float(row, "std_peak") for row in memory_rows])
    memory_y = np.asarray([_float(row, "memory_emotion_weight") for row in memory_rows])
    line_decay = axis_b.errorbar(decay_x, decay_y, xerr=decay_error, color="#1976BD", marker="o", linewidth=1.25, markersize=3.8, capsize=2.5, label="Emotion decay rate")
    axis_b_memory = axis_b.twinx()
    line_memory = axis_b_memory.errorbar(memory_x, memory_y, xerr=memory_error, color="#F57C00", marker="s", linewidth=1.25, markersize=3.8, capsize=2.5, label="Emotional-memory weight")
    axis_b.set_xlabel("Peak collective negative emotion $G_{peak}$", fontsize=7.2)
    axis_b.set_ylabel("Emotion decay rate $\\delta$", color="#1976BD", fontsize=7.2)
    axis_b_memory.set_ylabel("Emotional-memory weight $\\lambda$", color="#F57C00", fontsize=7.2)
    # Reserve a clear band above the highest estimates for the shared legend.
    axis_b.set_ylim(0.35, 0.45)
    axis_b_memory.set_ylim(0.195, 0.29)
    axis_b.tick_params(axis="y", colors="#1976BD", labelsize=6.4)
    axis_b_memory.tick_params(axis="y", colors="#F57C00", labelsize=6.4)
    axis_b.set_title("(b) Sensitivity of peak emotion to the\nhistorical-emotion-memory weight and emotion decay rate", loc="left", fontsize=8.2, fontweight="bold")
    axis_b.grid(axis="x", color="#E4E9ED", linewidth=0.55)
    axis_b.legend(
        [line_decay, line_memory],
        ["Emotion decay rate", "Emotional-memory weight"],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        fontsize=6.0,
        borderaxespad=0.0,
    )

    scales = sorted({int(_float(row, "num_agents")) for row in exp5_summary_rows})
    trajectory_grid = outer[1, 0].subgridspec(
        2,
        len(scales),
        height_ratios=[0.25, 0.75],
        hspace=0.03,
        wspace=0.13,
    )
    trajectory_heading = figure.add_subplot(trajectory_grid[0, :])
    trajectory_heading.axis("off")
    trajectory_heading.text(
        0.0,
        0.95,
        "(c) Trajectories of collective negative emotion\nacross agent-population sizes",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )
    trajectory_axes = [figure.add_subplot(trajectory_grid[1, index]) for index in range(len(scales))]
    for index, (axis, num_agents) in enumerate(zip(trajectory_axes, scales)):
        for scenario in SCENARIO_ORDER:
            group_label = f"{scenario}__n{num_agents}"
            group_rows = sorted([row for row in exp5_time_rows if row.get("group_label") == group_label], key=lambda row: _float(row, "step"))
            x = np.asarray([_float(row, "step") for row in group_rows])
            mean = np.asarray([_float(row, "mean_group_negative_emotion") for row in group_rows])
            std = np.asarray([_float(row, "std_group_negative_emotion") for row in group_rows])
            color = EXP1_SCENARIO_COLORS[scenario]
            axis.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.12, linewidth=0)
            axis.plot(x, mean, color=color, linewidth=1.05, label=SCENARIO_LABELS_EN[scenario])
        axis.set_title(f"{num_agents} agents", fontsize=6.8, fontweight="bold")
        axis.set_xlabel("Simulation step", fontsize=5.9)
        axis.set_xlim(0, 40)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#E4E9ED", linewidth=0.45)
        axis.tick_params(labelsize=5.5)
        if index:
            axis.tick_params(labelleft=False)
    trajectory_axes[0].set_ylabel("Collective negative emotion", fontsize=6.0)
    trajectory_handles, trajectory_labels = trajectory_axes[0].get_legend_handles_labels()
    figure.legend(
        trajectory_handles,
        trajectory_labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.27, 0.025),
        fontsize=5.8,
    )

    summary_grid = outer[1, 1].subgridspec(
        2,
        3,
        height_ratios=[0.25, 0.75],
        hspace=0.03,
        wspace=0.22,
    )
    summary_heading = figure.add_subplot(summary_grid[0, :])
    summary_heading.axis("off")
    summary_heading.text(
        0.0,
        0.95,
        "(d) Collective negative-emotion indicators\nacross agent-population sizes",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )
    summary_axes = [figure.add_subplot(summary_grid[1, index]) for index in range(3)]
    metric_specs = (
        ("mean_peak", "std_peak", "Peak collective\nnegative emotion"),
        ("mean_final_group_negative_emotion", "std_final_group_negative_emotion", "Final-round collective\nnegative emotion"),
        ("mean_final_high_negative_ratio", "std_final_high_negative_ratio", "Proportion of high-emotion\nagents"),
    )
    for axis_index, (axis, (metric, error_metric, title)) in enumerate(zip(summary_axes, metric_specs)):
        for scenario in SCENARIO_ORDER:
            scenario_rows = sorted([row for row in exp5_summary_rows if row.get("scenario_type") == scenario], key=lambda row: _float(row, "num_agents"))
            x = np.asarray([_float(row, "num_agents") for row in scenario_rows])
            values = np.asarray([_float(row, metric) for row in scenario_rows])
            errors = np.asarray([_float(row, error_metric) for row in scenario_rows])
            axis.errorbar(x, values, yerr=errors, color=EXP1_SCENARIO_COLORS[scenario], marker="o", linewidth=1.0, markersize=2.8, capsize=2.0, label=SCENARIO_LABELS_EN[scenario])
        axis.set_title(title, fontsize=6.3, fontweight="bold")
        axis.set_xlabel("Number of agents", fontsize=5.8)
        axis.set_xticks(scales)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#E4E9ED", linewidth=0.45)
        axis.tick_params(labelsize=5.3)
        if axis_index:
            axis.tick_params(labelleft=False)
    summary_axes[0].set_ylabel("Value", fontsize=6.0)
    summary_handles, summary_labels = summary_axes[0].get_legend_handles_labels()
    figure.legend(
        summary_handles,
        summary_labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.755, 0.025),
        fontsize=5.8,
    )

    bundle = _save_manuscript_bundle(figure, figures_dir / "figure_6_sensitivity_and_population_robustness", dpi=300)
    plt.close(figure)
    return {
        "figure_6_sensitivity_and_population_robustness": bundle["png"],
        "figure_6_sensitivity_and_population_robustness_tiff": bundle["tiff"],
        "figure_6_sensitivity_and_population_robustness_svg": bundle["svg"],
    }


def plot_experiment_5_agent_scale(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    """Update manuscript Figure 6 after the population-size experiment finishes."""

    _, exp5_summary_rows = _split_typed_rows(_read_csv(str(outputs["summary_path"])))
    _, exp5_time_rows = _split_typed_rows(_read_csv(str(outputs["time_series_path"])))
    _, exp4_rows = _split_typed_rows(_read_csv(EXPERIMENTS_DIR / "exp4_sensitivity_analysis" / "exp4_summary.csv"))
    return _plot_figure6_manuscript(exp4_rows, exp5_summary_rows, exp5_time_rows, figures_dir.parent / "exp4")


def _display_model_name(model: str) -> str:
    normalized = model.strip().lower()
    if normalized == "deepseek-v4-flash":
        return "DeepSeek-V4-Flash"
    if normalized == "gpt-5-nano":
        return "GPT-5-Nano"
    return model.strip()


def plot_experiment_6_model_robustness(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    """Create manuscript Figure 7 as one model-robustness composite."""

    _setup_matplotlib()
    _ensure_dir(figures_dir)
    _, summary_rows = _split_typed_rows(_read_csv(str(outputs["summary_path"])))
    _, time_rows = _split_typed_rows(_read_csv(str(outputs["time_series_path"])))
    if not summary_rows or not time_rows:
        return {}
    profile_ids = list(dict.fromkeys(row.get("profile_id", "") for row in summary_rows if row.get("profile_id")))
    profile_labels = {
        profile_id: _display_model_name(next((row.get("model", "") for row in summary_rows if row.get("profile_id") == profile_id), profile_id))
        for profile_id in profile_ids
    }

    figure = plt.figure(figsize=(8.1, 8.45))
    outer = figure.add_gridspec(2, 1, left=0.095, right=0.985, bottom=0.105, top=0.91, hspace=0.53)
    top_grid = outer[0].subgridspec(1, len(profile_ids), wspace=0.10)
    trajectory_axes = [figure.add_subplot(top_grid[0, index]) for index in range(len(profile_ids))]
    for axis_index, (axis, profile_id) in enumerate(zip(trajectory_axes, profile_ids)):
        for scenario in SCENARIO_ORDER:
            group_label = f"{profile_id}__{scenario}"
            group_rows = sorted([row for row in time_rows if row.get("group_label") == group_label], key=lambda row: _float(row, "step"))
            x = np.asarray([_float(row, "step") for row in group_rows])
            mean = np.asarray([_float(row, "mean_group_negative_emotion") for row in group_rows])
            std = np.asarray([_float(row, "std_group_negative_emotion") for row in group_rows])
            color = EXP1_SCENARIO_COLORS[scenario]
            axis.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.12, linewidth=0)
            axis.plot(x, mean, color=color, linewidth=1.25, label=SCENARIO_LABELS_EN[scenario])
        axis.set_title(profile_labels[profile_id], fontsize=8.2, fontweight="bold")
        axis.set_xlabel("Simulation step", fontsize=7.0)
        axis.set_xlim(0, 40)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#E4E9ED", linewidth=0.5)
        axis.tick_params(labelsize=6.4)
        if axis_index:
            axis.tick_params(labelleft=False)
    trajectory_axes[0].set_ylabel("Group mean negative emotion", fontsize=7.2)
    trajectory_axes[0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0.15, -0.21), fontsize=6.8)
    figure.text(0.02, 0.975, "(a) Trajectories of collective negative emotion across LLMs", fontsize=10.2, fontweight="bold", va="top")

    bottom_grid = outer[1].subgridspec(1, 3, wspace=0.34)
    summary_axes = [figure.add_subplot(bottom_grid[0, index]) for index in range(3)]
    metric_specs = (
        ("mean_peak", "std_peak", "Peak collective negative emotion"),
        ("mean_final_group_negative_emotion", "std_final_group_negative_emotion", "Final-round collective negative\nemotion"),
        ("mean_final_high_negative_ratio", "std_final_high_negative_ratio", "Final-round proportion of\nhigh-emotion agents"),
    )
    y_positions = np.arange(len(profile_ids), dtype=float)
    for axis_index, (axis, (metric, error_metric, title)) in enumerate(zip(summary_axes, metric_specs)):
        for scenario_index, scenario in enumerate(SCENARIO_ORDER):
            values = []
            errors = []
            for profile_id in profile_ids:
                row = next(item for item in summary_rows if item.get("profile_id") == profile_id and item.get("scenario_type") == scenario)
                values.append(_float(row, metric))
                errors.append(_float(row, error_metric))
            axis.errorbar(values, y_positions + (scenario_index - 1) * 0.16, xerr=errors, fmt="o", color=EXP1_SCENARIO_COLORS[scenario], markersize=4.0, capsize=2.4, label=SCENARIO_LABELS_EN[scenario])
        axis.set_title(title, fontsize=7.2, fontweight="bold")
        axis.set_xlim(0, 1)
        axis.set_ylim(len(profile_ids) - 0.5, -0.5)
        axis.set_yticks(y_positions)
        if axis_index == 0:
            axis.set_yticklabels([profile_labels[profile_id] for profile_id in profile_ids], fontsize=6.7)
        else:
            axis.set_yticklabels([])
        axis.grid(axis="y", color="#E4E9ED", linewidth=0.5)
        axis.tick_params(labelsize=6.3)
    summary_axes[0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=6.7)
    figure.text(0.02, 0.49, "(b) Collective-emotion indicators across LLMs", fontsize=10.2, fontweight="bold")

    bundle = _save_manuscript_bundle(figure, figures_dir / "figure_7_exp6_model_robustness", dpi=300)
    plt.close(figure)
    return {
        "figure_7_model_robustness": bundle["png"],
        "figure_7_model_robustness_tiff": bundle["tiff"],
        "figure_7_model_robustness_svg": bundle["svg"],
    }


def plot_experiment_4(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    """Update manuscript Figure 6 when population-size results are already available."""

    _, exp4_rows = _split_typed_rows(_read_csv(str(outputs["summary_path"])))
    exp5_dir = EXPERIMENTS_DIR / "exp5_agent_scale_robustness"
    _, exp5_summary_rows = _split_typed_rows(_read_csv(exp5_dir / "exp5_summary.csv"))
    _, exp5_time_rows = _split_typed_rows(_read_csv(exp5_dir / "exp5_time_series.csv"))
    return _plot_figure6_manuscript(exp4_rows, exp5_summary_rows, exp5_time_rows, figures_dir)


def _plot_exp2_manuscript_figure(
    time_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    raw_time_rows: list[dict[str, str]],
    figures_dir: Path,
) -> dict[str, str]:
    """Render manuscript Figure 4 with repeated exposure measured as memory emotion."""

    _setup_matplotlib()
    _ensure_dir(figures_dir)
    scenario_labels = {
        "risk_exaggeration": "Risk exaggeration",
        "selective_interpretation": "Selective interpretation",
        "emotional_wording": "Emotional wording",
    }
    level_labels = {"low": "Low", "medium": "Medium", "high": "High"}
    level_colors = {"low": "#FFA62B", "medium": "#2D8FE5", "high": "#FF2D2D"}

    figure = plt.figure(figsize=(10.6, 6.15))
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 1.03), hspace=0.56, wspace=0.68)
    recommendation_axis = figure.add_subplot(grid[0, :3])
    visibility_axis = figure.add_subplot(grid[0, 3:])
    repeat_axes = [figure.add_subplot(grid[1, 2 * index : 2 * index + 2]) for index in range(3)]

    x_positions = np.arange(len(SCENARIO_ORDER), dtype=float)
    bar_width = 0.22
    recommendation_rows = [
        row
        for row in summary_rows
        if row.get("mechanism_name") == "recommendation_strength"
        and row.get("mechanism_level") in LEVEL_ORDER
    ]
    for offset, level in zip((-bar_width, 0.0, bar_width), LEVEL_ORDER):
        level_rows = {
            row.get("scenario_type", ""): row
            for row in recommendation_rows
            if row.get("mechanism_level") == level
        }
        values = [
            _float(level_rows.get(scenario, {}), "mean_candidate_reach_total") / 10000.0
            for scenario in SCENARIO_ORDER
        ]
        errors = [
            _float(level_rows.get(scenario, {}), "std_candidate_reach_total") / 10000.0
            for scenario in SCENARIO_ORDER
        ]
        recommendation_axis.bar(
            x_positions + offset,
            values,
            width=bar_width,
            yerr=errors,
            color=level_colors[level],
            edgecolor="white",
            linewidth=0.7,
            error_kw={"ecolor": "#4D4D4D", "elinewidth": 0.8, "capsize": 2.6, "capthick": 0.8},
            label=level_labels[level],
        )
    recommendation_axis.set_title("(a) Recommendation amplification", loc="left", fontsize=10.5, fontweight="bold")
    recommendation_axis.set_xticks(x_positions)
    recommendation_axis.set_xticklabels([scenario_labels[item] for item in SCENARIO_ORDER], fontsize=8)
    recommendation_axis.set_ylabel("Candidate reach (10,000 exposures)", fontsize=9)
    recommendation_axis.grid(axis="y", color="#E5E9ED", linewidth=0.65)
    recommendation_axis.set_axisbelow(True)

    visibility_rows = [
        row
        for row in summary_rows
        if row.get("mechanism_name") == "visibility_strength"
        and row.get("mechanism_level") in LEVEL_ORDER
    ]
    for level in LEVEL_ORDER:
        level_rows = {
            row.get("scenario_type", ""): row
            for row in visibility_rows
            if row.get("mechanism_level") == level
        }
        values = [_float(level_rows.get(scenario, {}), "mean_unit_interaction_intensity") for scenario in SCENARIO_ORDER]
        errors = [_float(level_rows.get(scenario, {}), "std_unit_interaction_intensity") for scenario in SCENARIO_ORDER]
        visibility_axis.errorbar(
            x_positions,
            values,
            yerr=errors,
            marker="o",
            markersize=4.5,
            linewidth=1.45,
            capsize=2.8,
            color=level_colors[level],
            label=level_labels[level],
        )
    visibility_axis.set_title("(b) Network visibility", loc="left", fontsize=10.5, fontweight="bold")
    visibility_axis.set_xticks(x_positions)
    visibility_axis.set_xticklabels([scenario_labels[item] for item in SCENARIO_ORDER], fontsize=8)
    visibility_axis.set_ylabel("Interactions per user", fontsize=9)
    visibility_axis.grid(axis="y", color="#E5E9ED", linewidth=0.65)
    visibility_axis.set_axisbelow(True)

    repeat_rows = [
        row
        for row in time_rows
        if row.get("mechanism_name") == "repeat_strength"
        and row.get("mechanism_level") in LEVEL_ORDER
        and row.get("mean_memory_emotion") not in ("", None)
    ]
    for axis, scenario in zip(repeat_axes, SCENARIO_ORDER):
        for level in LEVEL_ORDER:
            level_rows = [
                row
                for row in repeat_rows
                if row.get("scenario_type") == scenario and row.get("mechanism_level") == level
            ]
            if not level_rows:
                continue
            steps, means = _series(level_rows, "mean_memory_emotion")
            standard_deviation_by_step = _step_std(
                raw_time_rows,
                "mean_memory_emotion",
                {
                    "scenario_type": scenario,
                    "mechanism_name": "repeat_strength",
                    "mechanism_level": level,
                },
            )
            standard_deviations = np.asarray(
                [standard_deviation_by_step.get(int(step), 0.0) for step in steps],
                dtype=float,
            )
            means_array = np.asarray(means, dtype=float)
            axis.fill_between(
                steps,
                np.maximum(0.0, means_array - standard_deviations),
                np.minimum(1.0, means_array + standard_deviations),
                color=level_colors[level],
                alpha=0.15,
                linewidth=0.0,
            )
            axis.plot(steps, means_array, color=level_colors[level], linewidth=1.55, label=level_labels[level])
        axis.set_title(scenario_labels[scenario], fontsize=9.3, fontweight="bold", pad=5)
        axis.set_xlabel("Simulation step", fontsize=8.5)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#E5E9ED", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8)
    repeat_axes[0].set_ylabel("Mean memory emotion", fontsize=9)
    repeat_axes[1].tick_params(labelleft=False)
    repeat_axes[2].tick_params(labelleft=False)
    figure.text(0.055, 0.466, "(c) Repeated exposure", fontsize=10.5, fontweight="bold")

    handles, labels = recommendation_axis.get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.005), fontsize=9)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.93)

    bundle = _save_manuscript_bundle(figure, figures_dir / "figure_4_exp2_platform_mechanisms", dpi=300)
    plt.close(figure)
    return {
        "figure_4_platform_mechanisms": bundle["png"],
        "figure_4_platform_mechanisms_tiff": bundle["tiff"],
        "figure_4_platform_mechanisms_svg": bundle["svg"],
    }


def plot_experiment_2(outputs: dict[str, object], figures_dir: Path) -> dict[str, str]:
    _ensure_dir(figures_dir)
    time_series_path = Path(str(outputs.get("time_series_path", "")))
    raw_rows, time_rows = _split_typed_rows(_read_csv(time_series_path))
    _, summary_rows = _split_typed_rows(_read_csv(str(outputs.get("summary_path", ""))))
    return _plot_exp2_manuscript_figure(time_rows, summary_rows, raw_rows, figures_dir)


def expand_experiment_1() -> dict[str, str]:
    exp_dir = EXPERIMENTS_DIR / "exp1_expression_styles"
    _, rows = _split_typed_rows(_read_csv(exp_dir / "exp1_time_series.csv"))
    return _plot_exp1_manuscript_figure(rows, FIGURES_DIR / "exp1") if rows else {}


def expand_experiment_2() -> dict[str, str]:
    exp_dir = EXPERIMENTS_DIR / "exp2_platform_main_effects"
    raw_rows, time_rows = _split_typed_rows(_read_csv(exp_dir / "exp2_time_series.csv"))
    _, summary_rows = _split_typed_rows(_read_csv(exp_dir / "exp2_summary.csv"))
    return _plot_exp2_manuscript_figure(time_rows, summary_rows, raw_rows, FIGURES_DIR / "exp2")


def expand_experiment_3() -> dict[str, str]:
    exp_dir = EXPERIMENTS_DIR / "exp3_scenario_platform_interaction"
    _, rows = _split_typed_rows(_read_csv(exp_dir / "exp3_summary.csv"))
    return _plot_experiment_3_summary(rows, FIGURES_DIR / "exp3") if rows else {}


def expand_experiment_4() -> dict[str, str]:
    exp4_dir = EXPERIMENTS_DIR / "exp4_sensitivity_analysis"
    exp5_dir = EXPERIMENTS_DIR / "exp5_agent_scale_robustness"
    _, exp4_rows = _split_typed_rows(_read_csv(exp4_dir / "exp4_summary.csv"))
    _, exp5_summary_rows = _split_typed_rows(_read_csv(exp5_dir / "exp5_summary.csv"))
    _, exp5_time_rows = _split_typed_rows(_read_csv(exp5_dir / "exp5_time_series.csv"))
    return _plot_figure6_manuscript(exp4_rows, exp5_summary_rows, exp5_time_rows, FIGURES_DIR / "exp4")


def expand_experiment_6() -> dict[str, str]:
    exp_dir = EXPERIMENTS_DIR / "exp6_model_robustness"
    return plot_experiment_6_model_robustness(
        {
            "summary_path": exp_dir / "exp6_combined_summary.csv",
            "time_series_path": exp_dir / "exp6_combined_time_series.csv",
        },
        FIGURES_DIR / "exp6",
    )


def generate_expanded_outputs() -> dict[str, str]:
    """Regenerate only the five quantitative figures used in the manuscript."""

    outputs: dict[str, str] = {}
    outputs.update(expand_experiment_1())
    outputs.update(expand_experiment_2())
    outputs.update(expand_experiment_3())
    outputs.update(expand_experiment_4())
    outputs.update(expand_experiment_6())
    return outputs
