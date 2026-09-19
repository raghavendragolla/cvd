import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

R = json.load(open("/tmp/claude-0/-home-claude/18c53616-59bf-5c3b-bdbc-96c0462aa827/scratchpad/results.json"))
OUT = "/tmp/claude-0/-home-claude/18c53616-59bf-5c3b-bdbc-96c0462aa827/scratchpad"

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e3e2df"
SURF = "#ffffff"
FOLDS = ["Cleveland", "Hungarian", "Switzerland", "VA"]
MODELS = ["LR", "RF", "XGB"]
MCOL = {"LR": S1, "RF": S2, "XGB": S3}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})

def style(ax, ylab, ymin, ymax, step):
    ax.set_ylabel(ylab, fontsize=8.5, color=INK)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

# ---------------------------------------------------------------- Fig 1
fig, ax = plt.subplots(figsize=(7.0, 2.7))
xs, labels = [], []
pos = 0.0
for fi, f in enumerate(FOLDS):
    for mi, m in enumerate(MODELS):
        v = R["folds"][f][m]
        a0, a1 = v["internal"]["auc"], v["transfer"]["auc"]
        lo, hi = v["transfer"]["auc_ci"]
        c = MCOL[m]
        ax.plot([pos, pos], [lo, hi], color=c, lw=1.4, alpha=0.30,
                solid_capstyle="round", zorder=2)
        ax.plot([pos, pos], [a0, a1], color=c, lw=2.0, alpha=0.55,
                solid_capstyle="round", zorder=3)
        ax.scatter([pos], [a0], s=26, facecolor=SURF, edgecolor=c, lw=1.6, zorder=4)
        ax.scatter([pos], [a1], s=34, color=c, edgecolor=SURF, lw=1.0, zorder=5)
        ax.annotate(f"{a1:.2f}", (pos, a1), textcoords="offset points",
                    xytext=(0, -11 if a1 < a0 else 7), ha="center",
                    fontsize=6.8, color=INK2, zorder=6)
        xs.append(pos); labels.append(m); pos += 1.0
    pos += 0.7

for fi, f in enumerate(FOLDS):
    centre = fi * 3.7 + 1.0
    ax.text(centre, 0.415, f, ha="center", fontsize=8.5, color=INK, weight="bold")

ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(pos - 0.9, 0.508, "chance", fontsize=6.5, color=MUTED, ha="right")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
style(ax, "ROC-AUC on the held-out cohort", 0.40, 1.0, 0.1)
ax.set_xlim(-0.8, pos - 0.9)

from matplotlib.lines import Line2D
leg = [Line2D([], [], marker="o", ls="", markerfacecolor=SURF, markeredgecolor=INK2,
              markeredgewidth=1.4, markersize=5, label="internal (5-fold CV)"),
       Line2D([], [], marker="o", ls="", color=INK2, markersize=6, label="external transfer")]
leg += [Line2D([], [], marker="s", ls="", color=MCOL[m], markersize=6, label=m) for m in MODELS]
ax.legend(handles=leg, ncol=5, frameon=False, fontsize=7, loc="upper center",
          bbox_to_anchor=(0.5, 1.16), handletextpad=0.4, columnspacing=1.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_discrimination.png", dpi=400, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2
fig, ax = plt.subplots(figsize=(7.0, 2.6))
states = [("frozen_eval", "Transferred (frozen)", S2),
          ("recal_intercept", "+ intercept recalibration", S1),
          ("recal_platt", "+ logistic recalibration", S3)]
w = 0.26
internal_mean = np.mean([R["folds"][f][m]["internal"]["ece"] for f in FOLDS for m in MODELS])
for si, (key, lab, col) in enumerate(states):
    vals = [np.mean([R["folds"][f][m][key]["ece"] for m in MODELS]) for f in FOLDS]
    x = np.arange(len(FOLDS)) + (si - 1) * (w + 0.018)
    ax.bar(x, vals, width=w, color=col, zorder=3, label=lab, linewidth=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.8, color=INK2, zorder=4)
ax.axhline(internal_mean, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.text(-0.44, internal_mean + 0.012, f"internal ECE ({internal_mean:.3f})",
        fontsize=6.6, color=MUTED, ha="left")
ax.set_xticks(np.arange(len(FOLDS))); ax.set_xticklabels(FOLDS, fontsize=8.5)
style(ax, "Expected Calibration Error", 0, 0.42, 0.1)
ax.legend(ncol=3, frameon=False, fontsize=7, loc="upper center",
          bbox_to_anchor=(0.5, 1.15), handletextpad=0.5, columnspacing=1.6)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_calibration.png", dpi=400, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 3
fig, ax = plt.subplots(figsize=(7.0, 2.5))
nf = R["noise_floor_kendall"]["mean"]
mix = [R["explanation"][f]["mix_driven"]["kendall"] for f in FOLDS]
mod = [R["explanation"][f]["model_driven"]["kendall"] if R["explanation"][f]["model_driven"] else np.nan
       for f in FOLDS]
w = 0.32
x = np.arange(len(FOLDS))
ax.bar(x - (w/2 + 0.01), mix, width=w, color=S1, zorder=3, linewidth=0,
       label="patient-mix driven (same model, new cohort)")
ax.bar(x + (w/2 + 0.01), [0 if np.isnan(v) else v for v in mod], width=w, color=S2,
       zorder=3, linewidth=0, label="model driven (same data, refitted model)")
for xi, v in zip(x - (w/2 + 0.01), mix):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK2, zorder=4)
for xi, v in zip(x + (w/2 + 0.01), mod):
    if np.isnan(v):
        ax.text(xi, 0.03, "n/a", ha="center", fontsize=6.8, color=MUTED, zorder=4, style="italic")
    else:
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK2, zorder=4)
ax.axhline(nf, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.text(3.48, nf + 0.022, f"explainer noise floor (τ = {nf:.2f})", fontsize=6.6,
        color=MUTED, ha="right")
ax.set_xticks(x); ax.set_xticklabels(FOLDS, fontsize=8.5)
style(ax, "Kendall τ between SHAP rankings", 0, 1.10, 0.2)
ax.legend(ncol=2, frameon=False, fontsize=7, loc="upper center",
          bbox_to_anchor=(0.5, 1.16), handletextpad=0.5, columnspacing=1.8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_explanation.png", dpi=400, bbox_inches="tight")
plt.close(fig)

print("figures written")
