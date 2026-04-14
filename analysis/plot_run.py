"""
plot_run.py
Generates all graphs for one training/test run and saves them as PNGs.

Usage:
    python plot_run.py \
        --stats  results/ep10000_decay0.9999/stats_10000_0.9999.pickle \
        --outdir results/ep10000_decay0.9999/graphs \
        --tag    ep10000_decay0.9999
"""

import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--stats",  required=True, help="Path to stats .pickle file")
parser.add_argument("--outdir", required=True, help="Directory to write PNGs into")
parser.add_argument("--tag",    required=True, help="Human-readable run label for titles")
args = parser.parse_args()

OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)
TAG = args.tag

with open(args.stats, "rb") as f:
    all_stats = pickle.load(f)

PHASES = [p for p in ("train", "test") if p in all_stats]
PHASE_COLORS = {"train": "#3266ad", "test": "#639922"}
PHASE_LABELS = {"train": "Training", "test": "Testing"}


# ── helpers ───────────────────────────────────────────────────────────────────

def rolling_avg(arr, window=100):
    out = np.convolve(arr, np.ones(window) / window, mode="valid")
    # pad front so length matches original
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, out])


def save(fig, name):
    path = OUTDIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}")


def base_fig(title, figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"{title}\n{TAG}", fontsize=11, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
    return fig, ax


# ── 1. Reward timeline (raw + smoothed, one subplot per phase) ────────────────

fig, axes = plt.subplots(1, len(PHASES), figsize=(6 * len(PHASES), 4), sharey=False)
if len(PHASES) == 1:
    axes = [axes]
fig.suptitle(f"Reward timeline — {TAG}", fontsize=11)

for ax, phase in zip(axes, PHASES):
    s = all_stats[phase]
    rewards = np.array(s["episode_rewards"])
    eps = np.arange(1, len(rewards) + 1)
    ax.plot(eps, rewards, color=PHASE_COLORS[phase], alpha=0.25, linewidth=0.6, label="per-episode")
    ax.plot(eps, rolling_avg(rewards, 100), color=PHASE_COLORS[phase], linewidth=1.5, label="rolling avg (100)")
    ax.set_title(PHASE_LABELS[phase], fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)

plt.tight_layout()
save(fig, "reward_timeline")

# ── 2. Smoothed reward only (cleaner view, both phases on one axes) ───────────

fig, ax = base_fig("Smoothed reward (rolling avg 100 ep)")
for phase in PHASES:
    rewards = np.array(all_stats[phase]["episode_rewards"])
    eps = np.arange(1, len(rewards) + 1)
    ax.plot(eps, rolling_avg(rewards, 100), color=PHASE_COLORS[phase],
            linewidth=1.8, label=PHASE_LABELS[phase])
ax.set_xlabel("Episode")
ax.set_ylabel("Rolling avg reward")
ax.legend(fontsize=9)
save(fig, "reward_smoothed")

# ── 3. Step distribution (histogram per phase) ───────────────────────────────

fig, axes = plt.subplots(1, len(PHASES), figsize=(5 * len(PHASES), 4), sharey=False)
if len(PHASES) == 1:
    axes = [axes]
fig.suptitle(f"Steps per episode — {TAG}", fontsize=11)

for ax, phase in zip(axes, PHASES):
    steps = np.array(all_stats[phase]["episode_steps"])
    ax.hist(steps, bins=40, color=PHASE_COLORS[phase], edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(steps), color="#e24b4a", linewidth=1.2, linestyle="--",
               label=f"mean {np.mean(steps):.0f}")
    ax.set_title(PHASE_LABELS[phase], fontsize=10)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Episodes")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save(fig, "step_distribution")

# ── 4. Q-table growth (training only) ────────────────────────────────────────

if "train" in all_stats and all_stats["train"]["table_sizes"]:
    fig, ax = base_fig("Q-table state count over training")
    sizes = all_stats["train"]["table_sizes"]
    ax.plot(np.arange(1, len(sizes) + 1), sizes,
            color=PHASE_COLORS["train"], linewidth=1.5)
    ax.fill_between(np.arange(1, len(sizes) + 1), sizes,
                    alpha=0.12, color=PHASE_COLORS["train"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Unique states")
    save(fig, "qtable_growth")

# ── 5. Outcome distribution (stacked bar, binned into 10 blocks) ─────────────

fig, axes = plt.subplots(1, len(PHASES), figsize=(5 * len(PHASES), 4), sharey=False)
if len(PHASES) == 1:
    axes = [axes]
fig.suptitle(f"Outcome distribution — {TAG}", fontsize=11)

WIN_COLOR   = "#639922"
LOSS_COLOR  = "#e24b4a"
TRUNC_COLOR = "#888780"

for ax, phase in zip(axes, PHASES):
    s = all_stats[phase]
    rewards = s["episode_rewards"]
    n = len(rewards)
    n_blocks = 10
    block_size = max(1, n // n_blocks)

    block_labels, wins, losses, truncs = [], [], [], []
    for b in range(n_blocks):
        start, end = b * block_size, min((b + 1) * block_size, n)
        pct = f"{int(b * 100 / n_blocks)}–{int((b+1) * 100 / n_blocks)}%"
        block_labels.append(pct)

        # outcomes are totals, so derive per-block from step slices using
        # episode-level outcome list if available, else fallback to overall
        ep_outcomes = s.get("episode_outcomes", None)
        if ep_outcomes:
            block_ep = ep_outcomes[start:end]
            w = sum(1 for o in block_ep if o == "win")
            l = sum(1 for o in block_ep if o == "loss")
            t = sum(1 for o in block_ep if o == "truncated")
        else:
            # No per-episode outcome list: distribute totals uniformly across blocks
            total = s["outcomes"]["win"] + s["outcomes"]["loss"] + s["outcomes"]["truncated"]
            block_n = end - start
            ratio = block_n / n if n > 0 else 0
            w = s["outcomes"]["win"] * ratio
            l = s["outcomes"]["loss"] * ratio
            t = s["outcomes"]["truncated"] * ratio
        wins.append(w)
        losses.append(l)
        truncs.append(t)

    x = np.arange(n_blocks)
    wins_a   = np.array(wins)
    losses_a = np.array(losses)
    truncs_a = np.array(truncs)
    ax.bar(x, wins_a,   color=WIN_COLOR,   label="Win",       edgecolor="white", linewidth=0.3)
    ax.bar(x, losses_a, color=LOSS_COLOR,  label="Loss",       bottom=wins_a, edgecolor="white", linewidth=0.3)
    ax.bar(x, truncs_a, color=TRUNC_COLOR, label="Truncated", bottom=wins_a + losses_a, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(block_labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(PHASE_LABELS[phase], fontsize=10)
    ax.set_xlabel("Training progress")
    ax.set_ylabel("Episodes")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save(fig, "outcome_distribution")

# ── 6. Move heatmap (bar chart, both phases side by side) ────────────────────

CARDS = ["rock", "paper", "scissors"]
CARD_COLORS = {"rock": "#3266ad", "paper": "#639922", "scissors": "#d85a30"}

fig, axes = plt.subplots(1, len(PHASES), figsize=(4 * len(PHASES), 4), sharey=False)
if len(PHASES) == 1:
    axes = [axes]
fig.suptitle(f"Move frequency — {TAG}", fontsize=11)

for ax, phase in zip(axes, PHASES):
    counts = [all_stats[phase]["move_counts"].get(c, 0) for c in CARDS]
    total_moves = sum(counts) or 1
    pcts = [c / total_moves * 100 for c in counts]
    bars = ax.bar(CARDS, pcts,
                  color=[CARD_COLORS[c] for c in CARDS],
                  edgecolor="white", linewidth=0.4)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(PHASE_LABELS[phase], fontsize=10)
    ax.set_ylabel("% of moves")
    ax.set_ylim(0, max(pcts) * 1.2 + 5)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save(fig, "move_heatmap")

# ── 7. Target heatmap (which opponent the agent challenged) ──────────────────

fig, axes = plt.subplots(1, len(PHASES), figsize=(4 * len(PHASES), 4), sharey=False)
if len(PHASES) == 1:
    axes = [axes]
fig.suptitle(f"Target frequency (opponent slot) — {TAG}", fontsize=11)

for ax, phase in zip(axes, PHASES):
    tc = all_stats[phase]["target_counts"]
    slots = sorted(tc.keys())
    counts = [tc[s] for s in slots]
    total_t = sum(counts) or 1
    pcts = [c / total_t * 100 for c in counts]
    labels = [f"P{s+1}" for s in slots]
    bars = ax.bar(labels, pcts, color="#7F77DD", edgecolor="white", linewidth=0.4)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(PHASE_LABELS[phase], fontsize=10)
    ax.set_ylabel("% of challenges")
    ax.set_ylim(0, max(pcts) * 1.2 + 5)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save(fig, "target_heatmap")

# ── 8. Decision split (pie per phase) ────────────────────────────────────────

fig, axes = plt.subplots(1, len(PHASES), figsize=(4 * len(PHASES), 4))
if len(PHASES) == 1:
    axes = [axes]
fig.suptitle(f"Decision split (random vs Q-table) — {TAG}", fontsize=11)

for ax, phase in zip(axes, PHASES):
    s = all_stats[phase]
    rand = s["random_decisions"]
    tbl  = s["table_decisions"]
    total_d = rand + tbl or 1
    sizes = [rand / total_d * 100, tbl / total_d * 100]
    labels = [f"Random\n{sizes[0]:.1f}%", f"Q-table\n{sizes[1]:.1f}%"]
    ax.pie(sizes, labels=labels, colors=["#888780", PHASE_COLORS[phase]],
           startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    ax.set_title(PHASE_LABELS[phase], fontsize=10)

plt.tight_layout()
save(fig, "decision_split")

print(f"\n  All graphs written to {OUTDIR}")