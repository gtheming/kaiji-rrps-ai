import sys, os

sys.path.insert(0, os.path.abspath(".."))
from environment_static.Q_learn import QLearnStatic
from environment_static.rrps_gym import StaticRRPSEnv
from tqdm import tqdm

env = StaticRRPSEnv(
    n_opponents=1,
    agent_budget={"paper_total": 3, "rock_total": 3, "scissors_total": 3},
    player_budget={"paper_total": 3, "rock_total": 3, "scissors_total": 3},
)
monty_hall = QLearnStatic(agent_name="monty_hall", env=env)

monty_hall.tabular_train(
    gamma=0.9, train_episodes=20_000, decay_rate=0.999, gui=False
)

rewards = []
wins = 0
losses = 0
truncations = 0

for _ in tqdm(range(10_000)):
    total_reward = 0
    for obs, reward, terminated, truncated, info in monty_hall.play_agent():
        total_reward += reward
    rewards.append(total_reward)
    if info["game_status"] == "victory":
        wins += 1
    elif info["game_status"] == "eliminated":
        losses += 1
    else:
        truncations += 1

total = len(rewards)
avg_reward = sum(rewards) / total
print(f"avg_reward: {avg_reward:.2f}")
print(f"win rate:   {wins/total*100:.1f}%  ({wins}/{total})")
print(f"loss rate:  {losses/total*100:.1f}%  ({losses}/{total})")
print(f"truncated:  {truncations/total*100:.1f}%  ({truncations}/{total})")
