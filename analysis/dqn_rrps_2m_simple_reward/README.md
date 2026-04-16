@dataclass
class RewardConfig:
win_matchup: float = 0
lose_matchup: float = 0
tie_matchup: float = 0
eliminated: float = -500
victory: float = 2000
invalid_move: float = -10
within_challenge_range: float = 0
approach_opponent: float = 0
