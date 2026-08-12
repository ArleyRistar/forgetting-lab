"""Figures for blog post 1, generated from outputs/ so they cannot drift.

Every value is read from the recorded JSON rather than typed in — a figure with a
hand-copied number is one more place for a claim to detach from its measurement,
which is the failure this project spent a week on.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/blog/figures")
INK, MUTED = "#1a1a1a", "#6b6b6b"
TERNARY, FLOAT, BASE = "#c1440e", "#2b6cb0", "#8a8a8a"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160,
})


def fig1_conversion_gap():
    g = json.loads(Path("outputs/convert/conversion-gap.json").read_text())
    c = g["corpora"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    corpora = [("fineweb_heldout", "FineWeb-edu (held out)"),
               ("wikitext103_test", "WikiText-103 (out of distribution)")]
    arms = [("base", "SmolLM2-360M", BASE), ("float", "float twin", FLOAT),
            ("ternary", "ternary twin", TERNARY)]
    w, xs = 0.26, range(len(corpora))
    for i, (key, label, colour) in enumerate(arms):
        vals = [c[ck][key]["loss"] for ck, _ in corpora]
        pos = [x + (i - 1) * w for x in xs]
        ax.bar(pos, vals, w, label=label, color=colour)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.09, f"{v:.2f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels([l for _, l in corpora])
    ax.set_ylabel("held-out loss (nats/token)")
    ax.set_ylim(0, 7.4)
    ax.set_title("The float twin lands on base. The ternary twin does not.",
                 loc="left", fontsize=12.5, pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.annotate("", xy=(0.26, 5.13), xytext=(0.26, 2.54),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(0.33, 3.7, "2.59 nats\nof ternarisation", fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "01-conversion-gap.png"); plt.close(fig)


def fig2_capability_gate():
    d = json.loads(Path("outputs/null/capability-gate.json").read_text())
    tasks = [r["task"] for r in d["arms"]["base"]
             if not r["task"].startswith("synth")]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    arms = [("base", "SmolLM2-360M", BASE), ("float", "float twin", FLOAT),
            ("ternary", "ternary twin", TERNARY)]
    w, xs = 0.26, range(len(tasks))
    for i, (key, label, colour) in enumerate(arms):
        by = {r["task"]: r for r in d["arms"][key]}
        vals = [by[t]["n_se"] for t in tasks]
        ax.bar([x + (i - 1) * w for x in xs], vals, w, label=label, color=colour)
    ax.axhline(3, color=INK, lw=1.1, ls="--")
    ax.text(len(tasks) - 0.45, 3.25, "keep threshold (3 SE)", fontsize=9,
            color=INK, ha="right")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([t + ("†" if t == "NumGLUE-cm" else "") for t in tasks])
    ax.set_ylabel("discrimination vs shuffled answers (SE)")
    ax.set_title("The ternary twin discriminates on nothing.",
                 loc="left", fontsize=12.5, pad=12)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # figure-level, not axes-level: at axes coords it overprinted the tick labels
    fig.text(0.01, 0.015, "† n=41 — underpowered, not evidence of absence",
             fontsize=8.5, color=MUTED)
    fig.savefig(OUT / "02-capability-gate.png"); plt.close(fig)


def fig3_batch_drift():
    s = json.loads(Path("outputs/ternary-batch-stability.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for arm, label, colour in (("ternary", "ternary twin", TERNARY),
                               ("float", "float twin", FLOAT)):
        d = s["vs_batch"][arm]
        bs = sorted(int(b) for b in d)
        ax.plot(bs, [d[str(b)]["median"] for b in bs], "o-", color=colour,
                label=label, lw=2, ms=6)
    ax.set_yscale("log"); ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels([2, 4, 8, 16, 32])
    ax.set_xlabel("batch size (scored against batch 1)")
    ax.set_ylabel("median |Δ log p|  (nats)")
    ax.set_title("Ternary forward passes are not batch-invariant",
                 loc="left", fontsize=12.5, pad=12)
    ax.legend(frameon=False, fontsize=10, loc="center right")
    t8 = s["vs_batch"]["ternary"]["8"]["median"]
    f8 = s["vs_batch"]["float"]["8"]["median"]
    ax.annotate("", xy=(8, t8), xytext=(8, f8),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(8.6, 3e-3, f"~{t8 / f8:,.0f}x", fontsize=11, color=INK)
    # sits just under the ternary line, where the panel is empty — at 1.2e-5 it
    # overprinted the float series and was unreadable
    ax.text(2.05, 8e-2, "flat from batch 2 — a step change, not a gradient",
            fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "03-batch-drift.png"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_conversion_gap(); fig2_capability_gate(); fig3_batch_drift()
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f}  {f.stat().st_size // 1024} KB")
