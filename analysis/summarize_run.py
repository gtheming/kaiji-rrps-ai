"""
summarize_run.py
Reads a stats pickle and prints fixed-width rows (one per phase) to stdout.
The shell script redirects stdout >> summary.txt.

Usage:
    python summarize_run.py \
        --stats   results/ep10000_decay0.9999/stats_10000_0.9999.pickle \
        --episodes 10000 \
        --decay    0.9999
"""

import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--stats",    required=True)
parser.add_argument("--episodes", required=True, type=int)
parser.add_argument("--decay",    required=True, type=float)
args = parser.parse_args()

with open(args.stats, "rb") as f:
    all_stats = pickle.load(f)

FMT = "%-8s %-10s %-8s %-12s %-8s %-8s %-8s %-8s %-8s %-14s %-12s %-12s"

for phase in ("train", "test"):
    if phase not in all_stats:
        continue
    s = all_stats[phase]

    rewards = s["episode_rewards"]
    n = len(rewards)
    if n == 0:
        continue

    total_decisions = s["random_decisions"] + s["table_decisions"]
    rand_pct  = s["random_decisions"] / total_decisions * 100 if total_decisions else 0
    table_pct = s["table_decisions"]  / total_decisions * 100 if total_decisions else 0

    total_time_s = sum(s["episode_times_ms"]) / 1000
    avg_ms       = sum(s["episode_times_ms"]) / n

    w = s["outcomes"]["win"]
    l = s["outcomes"]["loss"]
    t = s["outcomes"]["truncated"]

    table_states = s["table_sizes"][-1] if s["table_sizes"] else "n/a"

    print(FMT % (
        phase,
        args.episodes,
        args.decay,
        f"{np.mean(rewards):.1f}",
        f"{w/n*100:.1f}%",
        f"{l/n*100:.1f}%",
        f"{t/n*100:.1f}%",
        f"{rand_pct:.1f}%",
        f"{table_pct:.1f}%",
        table_states,
        f"{total_time_s:.1f}s",
        f"{avg_ms:.2f}ms",
    ))