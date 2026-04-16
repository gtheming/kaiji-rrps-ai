import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import ProgressBarCallback

from environment_dynamic.rrps_gym import RestrictedRPSEnv
import gym_core.visualizer as vis
import argparse

# Fixed obs cap — agent sees at most 4 nearest opponents within view_radius.
# Keeps obs shape (30,) constant so a trained model works across any n_opponents.
N_OBS_OPPONENTS = 4

env = RestrictedRPSEnv(
    n_opponents=10, stars=3, n_obs_opponents=N_OBS_OPPONENTS, grid_size=14
)

parser = argparse.ArgumentParser()
parser.add_argument("--file")
parser.add_argument("--train", action="store_true")
parser.add_argument("--gui", action="store_true")
args = parser.parse_args()
MODEL_PATH = args.file

train_flag = args.train
gui_flag = args.gui
if gui_flag and not train_flag:
    vis.init(grid_rows=env.grid_size, grid_cols=env.grid_size)

# ── Training ───────────────────────────────────────────────────────────────────
num_episodes = 2_000_000
gamma = 0.9

if train_flag:
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=gamma,
        target_update_interval=500,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=num_episodes, callback=ProgressBarCallback())
    model.save(MODEL_PATH)
    print(f"Saved to {MODEL_PATH}.zip")

# ── Evaluation ─────────────────────────────────────────────────────────────────

if not train_flag:
    model = DQN.load(MODEL_PATH, env=env)
    print(f"Loaded {MODEL_PATH}.zip")

    rewards = []
    wins = 0
    losses = 0
    truncations = 0
    rows = []
    n_eval_episodes = 10_000

    for episode in tqdm(range(n_eval_episodes)):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = False

        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            if gui_flag:
                vis.refresh(terminated, truncated, info)

        status = info["game_status"]
        agent = info["alive_player_dict"][0]
        rows.append(
            {
                "episode": episode,
                "game_status": status,
                "total_reward": total_reward,
                "turns": info["round_number"],
                "agent_stars": agent["stars_total"],
                "agent_cards_left": (
                    agent["rock_total"]
                    + agent["paper_total"]
                    + agent["scissors_total"]
                ),
                "opponents_alive": len(info["alive_player_dict"]) - 1,
            }
        )

        rewards.append(total_reward)
        if status == "victory":
            wins += 1
        elif status == "eliminated":
            losses += 1
        else:
            truncations += 1

    out_path = Path("analysis") / f"dynamic_results_{n_eval_episodes}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    total = len(rewards)
    print(f"avg_reward: {sum(rewards)/total:.2f}")
    print(f"win rate:   {wins/total*100:.1f}%  ({wins}/{total})")
    print(f"loss rate:  {losses/total*100:.1f}%  ({losses}/{total})")
    print(f"truncated:  {truncations/total*100:.1f}%  ({truncations}/{total})")
