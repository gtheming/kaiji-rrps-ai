from environment_static.rrps_gym import RestrictedRPSEnv
from tqdm import tqdm
import numpy as np
import argparse
import sys
import pickle
import time
from pathlib import Path
from gym_core.observation import Observation
from gym_core.cards import Card

parser = argparse.ArgumentParser()
parser.add_argument("mode", nargs="?", default="test",
                    choices=["train", "test"],
                    help="'train' to run training, omit or 'test' to evaluate")
parser.add_argument("--episodes", type=int,   default=20_000,
                    help="Number of training episodes (default: 20000)")
parser.add_argument("--decay",    type=float, default=0.999,
                    help="Epsilon decay rate per episode (default: 0.999)")
parser.add_argument("--outdir",   type=str,   default=".",
                    help="Directory to write pickle/qtable files into (default: cwd)")
parser.add_argument("--gui",      action="store_true",
                    help="Enable visualizer")
args = parser.parse_args()

train_flag = (args.mode == "train")
gui_flag = args.gui
if args.gui:
    import gym_core.visualizer as vis
if gui_flag:
    vis.init()

env = RestrictedRPSEnv(n_opponents=1, stars=3)
N_OPPONENTS = env.n_players - 1


# ========================================== State Hashing ==========================================

def obs_to_key(obs: Observation) -> tuple:
    agent = obs["player_dict"][0]
    opponents = sorted(
        ((pid, p) for pid, p in obs["player_dict"].items() if pid != 0),
        key=lambda x: x[0],
    )
    opponent_state = tuple(
        (
            p["stars_total"] > 0,
            p["rock_total"] > 0,
            p["paper_total"] > 0,
            p["scissors_total"] > 0,
        )
        for _, p in opponents
    )

    return (
        agent["stars_total"],
        agent["rock_total"],
        agent["paper_total"],
        agent["scissors_total"],
        opponent_state,
    )

# ========================================== stats helpers ==========================================
 
 
def _empty_stats(n_opponents: int) -> dict:
    """Return a zeroed-out stats dict for one phase (train or test)."""
    return {
        "episode_rewards":   [],
        "episode_steps":     [],
        "episode_times_ms":  [],
        "outcomes":          {"win": 0, "loss": 0, "truncated": 0},
        "random_decisions":  0,
        "table_decisions":   0,
        "move_counts":       {c.value: 0 for c in Card},
        "target_counts":     {i: 0 for i in range(n_opponents)},
        "table_sizes":       [],   # training only
    }
 
 
def _record_action(stats: dict, action: int, was_random: bool) -> None:
    """Decode an action int and update move/target/decision counters."""
    target_pid, card = env.action_resolve(action)
    opponent_slot = target_pid - 1
    stats["move_counts"][card.value] += 1
    if opponent_slot in stats["target_counts"]:
        stats["target_counts"][opponent_slot] += 1
    if was_random:
        stats["random_decisions"] += 1
    else:
        stats["table_decisions"] += 1
 
 
def _record_episode_end(
    stats: dict,
    ep_reward: float,
    ep_steps: int,
    ep_time_ms: float,
    game_status: str,
    table_size: int | None = None,
) -> None:
    stats["episode_rewards"].append(ep_reward)
    stats["episode_steps"].append(ep_steps)
    stats["episode_times_ms"].append(ep_time_ms)
 
    if game_status == "victory":
        stats["outcomes"]["win"] += 1
    elif game_status == "eliminated":
        stats["outcomes"]["loss"] += 1
    else:
        stats["outcomes"]["truncated"] += 1
 
    if table_size is not None:
        stats["table_sizes"].append(table_size)
 
 
def _print_stats_summary(stats: dict, phase: str) -> None:
    rewards = stats["episode_rewards"]
    total = len(rewards)
    if total == 0:
        return
 
    total_decisions = stats["random_decisions"] + stats["table_decisions"]
    rand_pct  = stats["random_decisions"] / total_decisions * 100 if total_decisions else 0
    table_pct = stats["table_decisions"]  / total_decisions * 100 if total_decisions else 0
    total_time_s = sum(stats["episode_times_ms"]) / 1000
    avg_time_ms  = sum(stats["episode_times_ms"]) / total
 
    w = stats["outcomes"]["win"]
    l = stats["outcomes"]["loss"]
    t = stats["outcomes"]["truncated"]
 
    print(f"\n{'─'*50}")
    print(f"  {phase.upper()} STATS  ({total} episodes)")
    print(f"{'─'*50}")
    print(f"  Total time:        {total_time_s:.1f}s")
    print(f"  Avg time/episode:  {avg_time_ms:.2f}ms")
    print(f"  Avg reward:        {sum(rewards)/total:.2f}")
    print(f"  Win rate:          {w/total*100:.1f}%  ({w}/{total})")
    print(f"  Loss rate:         {l/total*100:.1f}%  ({l}/{total})")
    print(f"  Truncated:         {t/total*100:.1f}%  ({t}/{total})")
    print(f"  Random decisions:  {rand_pct:.1f}%")
    print(f"  Table decisions:   {table_pct:.1f}%")
    print(f"  Move counts:       { {k: v for k, v in stats['move_counts'].items()} }")
    print(f"  Target counts:     { {f'P{k+1}': v for k, v in stats['target_counts'].items()} }")
    if stats["table_sizes"]:
        print(f"  Final table size:  {stats['table_sizes'][-1]:,} states")
    print(f"{'─'*50}\n")

# ========================================== Training ==========================================

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    """
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    Q_table = {}
    Q_update_counts = {}

    train_stats = _empty_stats(N_OPPONENTS)

    for _ in tqdm(range(num_episodes)):
        ep_start  = time.perf_counter()
        ep_reward = 0.0
        ep_steps  = 0

        start_obs, _ = env.reset()
        prev_state_key = obs_to_key(start_obs)
        if prev_state_key not in Q_update_counts:
            Q_update_counts[prev_state_key] = np.zeros(env.action_space.n)
        if prev_state_key not in Q_table:
            Q_table[prev_state_key] = np.zeros(env.action_space.n)
        while True:
            # initialize prev state keys if not already
            action = None

            # want random action w/ prob P = epsilon
            was_random = np.random.random() <= epsilon
            if was_random:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[prev_state_key])

            _record_action(train_stats, action, was_random)

            # transition to state s'
            new_obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_steps  += 1

            new_state_key = obs_to_key(new_obs)
            ## initalize state action if not already
            if new_state_key not in Q_update_counts:
                Q_update_counts[new_state_key] = np.zeros(env.action_space.n)
            if new_state_key not in Q_table:
                Q_table[new_state_key] = np.zeros(env.action_space.n)

            # calculate Q vals
            Q_old_update_counts = Q_update_counts[prev_state_key][action]
            Q_old = Q_table[prev_state_key][action]
            V_opt_old = np.max(Q_table[new_state_key])
            eta = 1 / (1 + Q_old_update_counts)
            Q_new = (1 - eta) * Q_old + eta * (reward + gamma * V_opt_old)

            # update table
            Q_table[prev_state_key][action] = Q_new
            Q_update_counts[prev_state_key][action] += 1

            if gui_flag:
                vis.refresh(terminated, truncated, info)
            # update epsilon and end or continue w/ new step as prev
            if terminated:
                epsilon *= decay_rate
                break
            else:
                prev_state_key = new_state_key

        ep_time_ms = (time.perf_counter() - ep_start) * 1000
        _record_episode_end(
            train_stats, ep_reward, ep_steps, ep_time_ms,
            game_status=info["game_status"],
            table_size=len(Q_table),
        )

    return Q_table, train_stats

def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)


"""
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
"""

num_episodes    = args.episodes
decay_rate      = args.decay
outdir          = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)
stats_filename  = str(outdir / f"stats_{num_episodes}_{decay_rate}.pickle")
qtable_filename = str(outdir / f"Q_table_{num_episodes}_{decay_rate}.pickle")

if train_flag:
    Q_table, train_stats = Q_learning(
        num_episodes=num_episodes,
        gamma=0.9,
        epsilon=1,
        decay_rate=decay_rate,
    )  # Run Q-learning

    # Save the Q-table dict to a file
    with open(qtable_filename, "wb") as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(stats_filename, "wb") as f:
        pickle.dump({"train": train_stats}, f, protocol=pickle.HIGHEST_PROTOCOL)
    _print_stats_summary(train_stats, "training")

    print(f"Completed", num_episodes, decay_rate)


if not train_flag:

    test_stats = _empty_stats(N_OPPONENTS)

    rewards = []
    wins = 0
    losses = 0
    truncations = 0

    with open(qtable_filename, "rb") as f:
        Q_table = pickle.load(f)
    print(
        f"Q_table: {len(Q_table)} states, {next(iter(Q_table.values())).shape}"
        " actions per state"
    )

    try:
        with open(qtable_filename, "rb") as f:
            all_stats = pickle.load(f)
    except FileNotFoundError:
        all_stats = {}


    for episode in tqdm(range(10000)):

        ep_start  = time.perf_counter()
        ep_reward = 0.0
        ep_steps  = 0

        obs, info = env.reset()
        total_reward = 0
        terminated = False
        while not terminated:
            state = obs_to_key(obs)

            try:
                action = np.random.choice(
                    env.action_space.n, p=softmax(Q_table[state])
                )  # Select action using softmax over Q-values
                was_random = False
            except KeyError:
                action = (
                    env.action_space.sample()
                )  # Fallback to random action if state not in Q-table
                was_random = True

            _record_action(test_stats, action, was_random)

            obs, reward, terminated, truncated, info = env.step(action)
            if gui_flag:
                vis.refresh(terminated, truncated, info)
            total_reward += reward

            ep_reward += reward
            ep_steps  += 1

        if info["game_status"] == "victory":
            wins += 1
        elif info["game_status"] == "eliminated":
            losses += 1
        else:
            truncations += 1

        # print("Total reward:", total_reward)
        rewards.append(total_reward)

        ep_time_ms = (time.perf_counter() - ep_start) * 1000
        _record_episode_end(
            test_stats, ep_reward, ep_steps, ep_time_ms,
            game_status=info["game_status"],
            table_size=None,
        )

    avg_reward = sum(rewards) / len(rewards)
    total = len(rewards)
    print(f"avg_reward: {avg_reward:.2f}")
    print(f"win rate:   {wins/total*100:.1f}%  ({wins}/{total})")
    print(f"loss rate:  {losses/total*100:.1f}%  ({losses}/{total})")
    print(f"truncated:  {truncations/total*100:.1f}%  ({truncations}/{total})")

    all_stats["test"] = test_stats
    with open(stats_filename, "wb") as f:
        pickle.dump(all_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    _print_stats_summary(test_stats, "testing")
    print(f"Updated stats → {stats_filename}")
