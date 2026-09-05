#!/usr/bin/env python3
"""Generate the main figures for the state-potential DiMA-PPI manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUPP = ROOT / "diffcormer_abbind_minexp" / "outputs" / "kbs_supplement"
PATHS = ROOT / "diffcormer_abbind_minexp" / "outputs" / "kbs_state_potential_path_audit"

BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#D97706"
RED = "#BE123C"
GREY = "#64748B"
DARK = "#172033"
LIGHT = "#E2E8F0"
PALE_BLUE = "#DBEAFE"
PALE_TEAL = "#CCFBF1"
PALE_ORANGE = "#FEF3C7"

MODEL_KEYS = ["pairwise", "soft_consistency", "hard_antisymmetry", "four_state_potential"]
MODEL_LABELS = {
    "pairwise": "Pairwise",
    "soft_consistency": "Soft",
    "hard_antisymmetry": "Hard anti.",
    "four_state_potential": "State potential",
}
MODEL_COLORS = {
    "pairwise": GREY,
    "soft_consistency": BLUE,
    "hard_antisymmetry": ORANGE,
    "four_state_potential": TEAL,
}

mpl.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.035,
})


def finish(fig, stem):
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png", dpi=600)
    plt.close(fig)


def clean(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#E8EDF3", lw=0.55, zorder=0)


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=7)


def node(ax, xy, text, face=PALE_BLUE, edge=BLUE, radius=0.085):
    patch = Circle(xy, radius, facecolor=face, edgecolor=edge, linewidth=1.4, zorder=3)
    ax.add_patch(patch)
    ax.text(*xy, text, ha="center", va="center", fontweight="bold", color=DARK, zorder=4)


def arrow(ax, start, end, text=None, color=DARK, rad=0.0, offset=(0, 0)):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.25,
                                connectionstyle=f"arc3,rad={rad}"))
    if text:
        mid = ((start[0] + end[0]) / 2 + offset[0],
               (start[1] + end[1]) / 2 + offset[1])
        ax.text(*mid, text, ha="center", va="center", color=color,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))


def fig_method():
    fig, axs = plt.subplots(2, 2, figsize=(7.48, 4.2))
    axs = axs.ravel()
    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axs[0]
    panel(ax, "A", "Edge regression")
    coords = {"A": (0.18, 0.22), "B": (0.82, 0.22), "C": (0.50, 0.78)}
    for key, xy in coords.items():
        node(ax, xy, key, face="#F1F5F9", edge=GREY)
    arrow(ax, (0.26, 0.25), (0.73, 0.25), r"$f(A,B)$", GREY, offset=(0, .07))
    arrow(ax, (0.75, 0.29), (0.55, 0.69), r"$f(B,C)$", GREY, offset=(.08, .01))
    arrow(ax, (0.44, 0.69), (0.24, 0.29), r"$f(C,A)$", GREY, offset=(-.08, .01))
    ax.text(.5, .04, "Edges need not form a state function", ha="center", color=RED)

    ax = axs[1]
    panel(ax, "B", "State-potential learning")
    for key, xy in coords.items():
        node(ax, xy, key, face=PALE_TEAL, edge=TEAL)
        ax.text(xy[0], xy[1] - .15, rf"$\Phi({key})$", ha="center", color=TEAL)
    arrow(ax, (0.27, .23), (.72, .23), r"$\Phi(B)-\Phi(A)$", TEAL, offset=(0, .08))
    arrow(ax, (.74, .29), (.55, .68), color=TEAL)
    arrow(ax, (.45, .68), (.25, .29), color=TEAL)
    ax.text(.5, .04, r"Discrete gradient: $\sum_C \hat y_{ij}=0$", ha="center", color=TEAL)

    ax = axs[2]
    panel(ax, "C", "PPI bound/free decomposition")
    positions = {"A$_f$": (.2, .2), "B$_f$": (.8, .2), "A$_b$": (.2, .74), "B$_b$": (.8, .74)}
    for key, xy in positions.items():
        node(ax, xy, key, face=PALE_ORANGE, edge=ORANGE, radius=.075)
    arrow(ax, (.29, .2), (.71, .2), r"$\Delta E_f$", ORANGE, offset=(0, .07))
    arrow(ax, (.29, .74), (.71, .74), r"$\Delta E_b$", ORANGE, offset=(0, .07))
    arrow(ax, (.2, .29), (.2, .65), color=GREY)
    arrow(ax, (.8, .29), (.8, .65), color=GREY)
    ax.text(.5, .43, r"$\widehat{\Delta\Delta G}=\Delta E_b-\Delta E_f$",
            ha="center", va="center", fontweight="bold", color=DARK,
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec=LIGHT))

    ax = axs[3]
    panel(ax, "D", "Canonical state identity")
    node(ax, (.18, .5), "A", face="#F1F5F9", edge=GREY)
    node(ax, (.50, .5), "B", face=PALE_TEAL, edge=TEAL)
    node(ax, (.82, .5), "C", face="#F1F5F9", edge=GREY)
    arrow(ax, (.27, .5), (.41, .5), color=TEAL)
    arrow(ax, (.59, .5), (.73, .5), color=TEAL)
    box = FancyBboxPatch((.18, .14), .64, .16, boxstyle="round,pad=.03",
                         facecolor="#F8FAFC", edgecolor=LIGHT)
    ax.add_patch(box)
    ax.text(.5, .22, "one cache key → one input → one $\\Phi(B)$",
            ha="center", va="center", color=DARK)
    ax.text(.5, .82, "same state reused across paths", ha="center", color=TEAL)

    fig.subplots_adjust(wspace=.22, hspace=.30)
    finish(fig, "fig1_state_potential_concept")


def ecdf(values):
    x = np.sort(np.maximum(np.asarray(values, dtype=float), 1e-9))
    return x, np.arange(1, len(x) + 1) / len(x)


def fig_integrability():
    summary = pd.read_csv(PATHS / "real_mutation_integrability_table.csv").set_index("variant")
    paths = pd.read_csv(PATHS / "real_mutation_path_predictions.csv")
    cycles = pd.read_csv(PATHS / "real_mutation_four_state_cycles.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.48, 5.1))

    audits = [
        ("reverse_error_mean", "Reverse", "#93C5FD"),
        ("path_error_mean", "Path", TEAL),
        ("triangle_cycle_error_mean", "Triangle", ORANGE),
        ("four_state_cycle_error_mean", "4-state cycle", RED),
    ]
    x = np.arange(4)
    width = .19
    for j, (column, label, color) in enumerate(audits):
        vals = [max(float(summary.loc[k, column]), 1e-9) for k in MODEL_KEYS]
        axs[0, 0].bar(x + (j - 1.5) * width, vals, width, color=color,
                      edgecolor="white", linewidth=.35, label=label, zorder=2)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xticks(x, [MODEL_LABELS[k] for k in MODEL_KEYS], rotation=18, ha="right")
    axs[0, 0].set_ylabel("Mean absolute error (kcal mol$^{-1}$)")
    axs[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    panel(axs[0, 0], "A", "Reverse consistency is weaker than integrability")
    clean(axs[0, 0])

    for key in MODEL_KEYS:
        xx, yy = ecdf(paths.loc[paths.variant == key, "path_error"])
        axs[0, 1].plot(xx, yy, color=MODEL_COLORS[key], label=MODEL_LABELS[key])
    axs[0, 1].set_xscale("log")
    axs[0, 1].axvline(.1, color=DARK, ls="--", lw=.75)
    axs[0, 1].set_xlabel(r"Three-state path error (kcal mol$^{-1}$)")
    axs[0, 1].set_ylabel("Cumulative fraction")
    axs[0, 1].legend(frameon=False, loc="lower right")
    panel(axs[0, 1], "B", "Real mutation paths")
    clean(axs[0, 1], grid=False)

    for key in MODEL_KEYS:
        xx, yy = ecdf(cycles.loc[cycles.variant == key, "cycle_error"])
        axs[1, 0].plot(xx, yy, color=MODEL_COLORS[key], label=MODEL_LABELS[key])
    axs[1, 0].set_xscale("log")
    axs[1, 0].axvline(.1, color=DARK, ls="--", lw=.75)
    axs[1, 0].set_xlabel(r"Four-state cycle circulation (kcal mol$^{-1}$)")
    axs[1, 0].set_ylabel("Cumulative fraction")
    panel(axs[1, 0], "C", "Closed mutation cycles")
    clean(axs[1, 0], grid=False)

    focus = paths[paths.variant.isin(["hard_antisymmetry", "four_state_potential"])]
    rng = np.random.default_rng(20260822)
    for key in ["hard_antisymmetry", "four_state_potential"]:
        frame = focus[focus.variant == key]
        if len(frame) > 1200:
            frame = frame.iloc[rng.choice(len(frame), 1200, replace=False)]
        direct = frame.direct_prediction.to_numpy()
        composed = (frame.step_one_prediction + frame.step_two_prediction).to_numpy()
        axs[1, 1].scatter(direct, composed, s=7, alpha=.26, color=MODEL_COLORS[key],
                          label=MODEL_LABELS[key], rasterized=True, linewidth=0)
    lim = (-7.5, 7.5)
    axs[1, 1].plot(lim, lim, color=DARK, ls="--", lw=.8)
    axs[1, 1].set_xlim(lim)
    axs[1, 1].set_ylim(lim)
    axs[1, 1].set_aspect("equal", adjustable="box")
    axs[1, 1].set_xlabel(r"Direct $\hat y_{A\to C}$")
    axs[1, 1].set_ylabel(r"Composed $\hat y_{A\to B}+\hat y_{B\to C}$")
    axs[1, 1].legend(frameon=False, loc="upper left")
    panel(axs[1, 1], "D", "Hard antisymmetry remains path dependent")
    clean(axs[1, 1], grid=False)

    fig.subplots_adjust(wspace=.33, hspace=.40)
    finish(fig, "fig2_integrability_audit")


def fig_predictive_value():
    models = pd.read_csv(SUPP / "state_potential_model_comparison.csv")
    boot = pd.read_csv(SUPP / "state_potential_canonical_paired_bootstrap.csv")
    short = ["Pairwise", "Reverse labels", "Soft consistency", "Hard antisymmetry",
             "Potential, no state", "Four-state potential"]
    colors = [GREY, "#60A5FA", BLUE, ORANGE, "#5EEAD4", TEAL]
    y_models = np.arange(len(models))[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(7.48, 2.55), gridspec_kw={"width_ratios": [1, 1, .92]})
    for ax, mean_col, sd_col, ylabel, letter in [
        (axs[0], "pearson_mean", "pearson_sd", "Fold-mean PCC", "A"),
        (axs[1], "rmse_mean", "rmse_sd", "RMSE (kcal mol$^{-1}$)", "B"),
    ]:
        for yi, mean, sd, color in zip(y_models, models[mean_col], models[sd_col], colors):
            ax.errorbar(mean, yi, xerr=sd, fmt="o", color=color, ecolor=color,
                        capsize=2.2, zorder=3)
        ax.set_yticks(y_models, short)
        ax.set_xlabel(ylabel)
        panel(ax, letter, "Matched S1129 models")
        clean(ax)
    axs[0].set_xlim(.52, .79)
    axs[1].set_xlim(1.45, 2.17)

    order = ["PCC", "RMSE", "MAE"]
    b = boot.set_index("metric").loc[order]
    y = np.arange(3)[::-1]
    diff = b.difference_four_minus_no_state.to_numpy()
    lo = b.ci95_low.to_numpy()
    hi = b.ci95_high.to_numpy()
    axs[2].errorbar(diff, y, xerr=np.vstack([diff - lo, hi - diff]), fmt="o",
                    color=TEAL, ecolor=TEAL, capsize=2.5, zorder=3)
    axs[2].axvline(0, color=DARK, ls="--", lw=.75)
    axs[2].set_yticks(y, order)
    axs[2].set_xlabel("Four-state minus no-state")
    panel(axs[2], "C", "Complex bootstrap (95% CI)")
    clean(axs[2])
    fig.subplots_adjust(wspace=.43)
    finish(fig, "fig3_predictive_value")


def fig_strict_boundary():
    data = pd.read_csv(SUPP / "canonical_state_potential_strict_table.csv")
    boot = json.loads((SUPP / "canonical_state_potential_strict_bootstrap.json").read_text())
    labels = ["Backbone", "Soft", "Hard anti.", "State potential"]
    model_names = ["Unconstrained backbone", "Soft consistency", "Hard antisymmetry",
                   "Canonical state potential"]
    colors = [GREY, BLUE, ORANGE, TEAL]
    fig, axs = plt.subplots(1, 3, figsize=(7.48, 2.55), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    splits = [("sequence_cluster", "Sequence"), ("interface_structure", "Interface")]
    x = np.arange(4)
    for ax, metric, ylabel, letter in [
        (axs[0], "PCC", "Five-seed ensemble PCC", "A"),
        (axs[1], "RMSE", "RMSE (kcal mol$^{-1}$)", "B"),
    ]:
        for j, (split_key, split_label) in enumerate(splits):
            frame = data[data.split == split_key].set_index("model").loc[model_names]
            ax.plot(x, frame[metric], marker="o", color=[BLUE, RED][j],
                    label=split_label, zorder=3)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        panel(ax, letter, f"{metric} under distribution shift")
        clean(ax)

    ax = axs[2]
    entries = []
    names = []
    for split_key, split_label in splits:
        for comp_key, comp_label in [("canonical_minus_backbone", "vs backbone"),
                                     ("canonical_minus_soft", "vs soft"),
                                     ("canonical_minus_hard", "vs hard")]:
            row = next(v for v in boot[split_key][comp_key] if v["metric"] == "PCC")
            entries.append(row)
            short_split = "Seq" if split_label == "Sequence" else "Int"
            names.append(f"{short_split} {comp_label.replace('vs ', '−')}")
    y = np.arange(len(entries))[::-1]
    d = np.array([v["difference"] for v in entries])
    lo = np.array([v["ci95_low"] for v in entries])
    hi = np.array([v["ci95_high"] for v in entries])
    point_colors = [BLUE] * 3 + [RED] * 3
    for yi, di, li, hi_i, color in zip(y, d, lo, hi, point_colors):
        ax.errorbar(di, yi, xerr=[[di - li], [hi_i - di]], fmt="o", color=color,
                    ecolor=color, capsize=2.2, zorder=3)
    ax.axvline(0, color=DARK, ls="--", lw=.75)
    ax.set_yticks(y, names)
    ax.set_xlabel("State-potential minus control PCC")
    panel(ax, "C", "PCC effects (95% CI)")
    clean(ax)
    fig.subplots_adjust(wspace=.44)
    finish(fig, "fig4_strict_boundary")


if __name__ == "__main__":
    fig_method()
    fig_integrability()
    fig_predictive_value()
    fig_strict_boundary()
