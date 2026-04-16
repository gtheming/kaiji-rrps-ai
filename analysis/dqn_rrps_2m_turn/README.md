num_episodes = 2_000_000
gamma = 0.9

class RewardConfig:
win_matchup: float = 100
lose_matchup: float = -100
tie_matchup: float = 10
eliminated: float = -500
victory: float = 2000
invalid_move: float = -10
within_challenge_range: float = 1
approach_opponent: float = 0.5
