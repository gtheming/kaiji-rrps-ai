from __future__ import annotations

import random
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gym_core.cards import Card
from gym_core.info import Info, GameStatus
from gym_core.matchup_dict import MatchupDict
from gym_core.observation import PlayerObs, NavObservation as Observation
from gym_core.player import PlayerDict, PlayerID, Player
from gym_core.reward_config import RewardConfig as _BaseRewardConfig
from gym_core.rrps_gym import RRPSEnvCore


@dataclass
class RewardConfig(_BaseRewardConfig):
    pass


def resolve(m1: Card, m2: Card) -> int:
    """Returns 1 if m1 wins, -1 if m2 wins, 0 on tie."""
    wins_against = {
        Card.rock: Card.scissors,
        Card.paper: Card.rock,
        Card.scissors: Card.paper,
    }
    if m1 == m2:
        return 0
    return 1 if wins_against[m1] == m2 else -1


class RestrictedRPSEnv(RRPSEnvCore):
    """
    A single-agent Gymnasium environment for Restricted Rock Paper Scissors.

    The *agent* controls player 0; all other players act as BasicPlayers on a 2D grid.
    Players can only challenge opponents within challenge_radius squares (Chebyshev
    distance). Opponents move toward the nearest player when no target is in range.

    Action (Discrete 7):
        0 = Move Up    (y - 1)
        1 = Move Down  (y + 1)
        2 = Move Left  (x - 1)
        3 = Move Right (x + 1)
        4 = ROCK
        5 = PAPER
        6 = SCISSORS
    """

    metadata = {"render_modes": ["human"]}

    _MOVE_ACTIONS = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
    }
    _RPS_ACTIONS = {4: Card.rock, 5: Card.paper, 6: Card.scissors}

    def __init__(
        self,
        n_opponents: int = 3,
        stars: int = 3,
        budget: int = 4,
        grid_size: int = 20,
        challenge_radius: int = 1,
        max_turns: int = 2000,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()
        self.n_opponents = n_opponents
        self.initial_stars = stars
        self.initial_budget = budget
        self.grid_size = grid_size
        self.challenge_radius = challenge_radius
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.reward_config = reward_config or RewardConfig()

        max_stars = stars + n_opponents * stars
        g = grid_size - 1
        player_space = spaces.Dict(
            {
                "player_id": spaces.Discrete(n_opponents + 1),
                "stars_total": spaces.Discrete(max_stars + 1),
                "budget": spaces.Dict(
                    {
                        "rock_total": spaces.Discrete(2),
                        "paper_total": spaces.Discrete(2),
                        "scissors_total": spaces.Discrete(2),
                    }
                ),
                "position": spaces.Tuple(
                    (spaces.Discrete(g + 1), spaces.Discrete(g + 1))
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "agent": player_space,
                "opponent": player_space,
                "opponents_alive": spaces.Discrete(n_opponents + 1),
            }
        )
        self.action_space = spaces.Discrete(7)

        self.player_dict: PlayerDict = {}
        self.still_playing_dict: PlayerDict = {}

    # ── private helpers ───────────────────────────────────────────────────────

    def _random_position(self) -> tuple[int, int]:
        return (
            int(self.np_random.integers(0, self.grid_size)),
            int(self.np_random.integers(0, self.grid_size)),
        )

    def _initialize_players(self):
        self.player_dict = {
            i: {
                "rock_total": self.initial_budget,
                "paper_total": self.initial_budget,
                "scissors_total": self.initial_budget,
                "stars_total": self.initial_stars,
                "position": self._random_position(),
            }
            for i in range(self.n_opponents + 1)
        }
        self.still_playing_dict = self.player_dict.copy()

    def _total_cards(self, player: Player) -> int:
        return player["rock_total"] + player["paper_total"] + player["scissors_total"]

    def _has_cards(self, pid: PlayerID) -> bool:
        return self._total_cards(self.player_dict[pid]) > 0

    def _is_alive(self, pid: PlayerID) -> bool:
        return self.player_dict[pid]["stars_total"] > 0

    def _update_playing(self):
        new_playing = {}
        for pid, player in self.still_playing_dict.items():
            if self._total_cards(player) > 0 and player["stars_total"] > 0:
                new_playing[pid] = player
        new_playing[0] = self.player_dict[0]  # agent death handled at step end
        self.still_playing_dict = new_playing

    def _alive_opponents(self) -> list[PlayerID]:
        return [pid for pid in self.still_playing_dict if pid != 0]

    def _in_range(self, a: PlayerID, b: PlayerID) -> bool:
        return (
            chebyshev(self.player_dict[a]["position"], self.player_dict[b]["position"])
            <= self.challenge_radius
        )

    def _can_move(self, pos: tuple[int, int], delta: tuple[int, int]) -> bool:
        x, y = pos[0] + delta[0], pos[1] + delta[1]
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _nearest(self, pid: PlayerID, candidates: list[PlayerID]) -> PlayerID | None:
        if not candidates:
            return None
        pos = self.player_dict[pid]["position"]
        return min(candidates, key=lambda p: chebyshev(pos, self.player_dict[p]["position"]))

    def _toward_direction(self, pid: PlayerID, target_pid: PlayerID) -> Direction:
        pos = self.player_dict[pid]["position"]
        target = self.player_dict[target_pid]["position"]
        dx = int(np.sign(target[0] - pos[0]))
        dy = int(np.sign(target[1] - pos[1]))
        best_score = min(abs(d.value[0] - dx) + abs(d.value[1] - dy) for d in Direction)
        best = [d for d in Direction if abs(d.value[0] - dx) + abs(d.value[1] - dy) == best_score]
        return random.choice(best)

    def _select_move(self, pid: PlayerID) -> Card:
        available = [c for c in Card if self.still_playing_dict[pid][c.value] > 0]
        return random.choice(available)

    def _resolve_matchups(self, matchup_dict: MatchupDict):
        for (pid1, pid2), (card1, card2) in matchup_dict.items():
            result = resolve(card1, card2)
            if result == 1:
                self.player_dict[pid1]["stars_total"] += 1
                self.player_dict[pid2]["stars_total"] -= 1
            elif result == -1:
                self.player_dict[pid2]["stars_total"] += 1
                self.player_dict[pid1]["stars_total"] -= 1
            self.player_dict[pid1][card1.value] -= 1
            self.player_dict[pid2][card2.value] -= 1

    def _player_obs(self, pid: PlayerID) -> PlayerObs:
        p = self.player_dict[pid]
        return {
            "player_id": pid,
            "stars_total": p["stars_total"],
            "budget": {
                "rock_total": int(p["rock_total"] > 0),
                "paper_total": int(p["paper_total"] > 0),
                "scissors_total": int(p["scissors_total"] > 0),
            },
            "position": p["position"],
        }

    def _null_opponent_obs(self) -> PlayerObs:
        return {
            "player_id": 0,
            "stars_total": 0,
            "budget": {"rock_total": 0, "paper_total": 0, "scissors_total": 0},
            "position": (0, 0),
        }

    def _get_obs(self) -> Observation:
        alive = self._alive_opponents()
        if alive:
            in_range = [op for op in alive if self._in_range(0, op)]
            nearest = self._nearest(0, in_range or alive)
            opponent_obs = self._player_obs(nearest)
        else:
            opponent_obs = self._null_opponent_obs()
        return {
            "agent": self._player_obs(0),
            "opponent": opponent_obs,
            "opponents_alive": len(alive),
        }

    def _get_info(
        self,
        initial_alive_player_dict: PlayerDict,
        game_status: GameStatus,
        challenge_table: None = None,
        matchup_dict: MatchupDict | None = None,
    ) -> Info:
        return {
            "challenge_table": challenge_table,
            "matchup_dict": matchup_dict,
            "alive_player_dict": self.still_playing_dict,
            "round_number": self._turn,
            "initial_alive_player_dict": initial_alive_player_dict,
            "game_status": game_status,
        }

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._initialize_players()
        self._turn = 0
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False
        truncated = False
        game_status: GameStatus = "playing"
        matchup_dict: MatchupDict = {}

        initial_alive_player_dict: PlayerDict = self.still_playing_dict.copy()
        agent = self.player_dict[0]
        agent_card: Card | None = None

        # ── Agent action ──────────────────────────────────────────────────────
        if action in self._MOVE_ACTIONS:
            dx, dy = self._MOVE_ACTIONS[action]
            pos = agent["position"]
            if self._can_move(pos, (dx, dy)):
                agent["position"] = (pos[0] + dx, pos[1] + dy)
            else:
                reward += self.reward_config.invalid_move
        else:
            card = self._RPS_ACTIONS[action]
            in_range = [op for op in self._alive_opponents() if self._in_range(0, op)]
            if not in_range or agent[card.value] == 0:
                reward += self.reward_config.invalid_move
            else:
                agent_card = card

        # ── Move opponents toward nearest ─────────────────────────────────────
        all_pids = list(self.still_playing_dict.keys())
        for pid in all_pids:
            if pid == 0:
                continue
            others = [q for q in all_pids if q != pid]
            if not others:
                direction = random.choice(list(Direction))
            else:
                target = self._nearest(pid, others)
                direction = self._toward_direction(pid, target)
            pos = self.player_dict[pid]["position"]
            if self._can_move(pos, direction.value):
                self.player_dict[pid]["position"] = (
                    pos[0] + direction.value[0],
                    pos[1] + direction.value[1],
                )

        # ── Build matchups from in-range pairs ────────────────────────────────
        alive_pids = list(self.still_playing_dict.keys())
        matched: set[PlayerID] = set()

        for i, pid1 in enumerate(alive_pids):
            if pid1 in matched:
                continue
            for pid2 in alive_pids[i + 1:]:
                if pid2 in matched:
                    continue
                if not self._in_range(pid1, pid2):
                    continue
                if self._total_cards(self.player_dict[pid1]) == 0 or self._total_cards(self.player_dict[pid2]) == 0:
                    continue
                if (pid1 == 0 or pid2 == 0) and agent_card is None:
                    continue
                c1 = agent_card if pid1 == 0 else self._select_move(pid1)
                c2 = agent_card if pid2 == 0 else self._select_move(pid2)
                matchup_dict[(pid1, pid2)] = (c1, c2)
                matched.add(pid1)
                matched.add(pid2)
                break

        # ── Resolve matchups ──────────────────────────────────────────────────
        agent_stars_before = agent["stars_total"]
        self._resolve_matchups(matchup_dict)
        agent_stars_after = agent["stars_total"]

        if any(0 in pair for pair in matchup_dict):
            delta = agent_stars_after - agent_stars_before
            if delta > 0:
                reward += self.reward_config.win_matchup
            elif delta < 0:
                reward += self.reward_config.lose_matchup
            else:
                reward += self.reward_config.tie_matchup

        # ── Update alive dict ─────────────────────────────────────────────────
        self._update_playing()

        # ── Termination ───────────────────────────────────────────────────────
        agent_no_cards = self._total_cards(agent) == 0
        agent_no_stars = agent["stars_total"] <= 0

        if agent_no_cards or agent_no_stars:
            terminated = True
            game_status = "victory" if agent["stars_total"] >= self.initial_stars else "eliminated"
            reward += self.reward_config.victory if game_status == "victory" else self.reward_config.eliminated
        elif len(self.still_playing_dict) == 1:
            terminated = True
            game_status = "victory"
            reward += self.reward_config.victory
        elif self._turn >= self.max_turns:
            terminated = True
            game_status = "eliminated"
            reward += self.reward_config.eliminated

        obs = self._get_obs()
        info = self._get_info(initial_alive_player_dict, game_status, None, matchup_dict)
        self._turn += 1
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        p = self.player_dict[0]
        print(
            f"[Agent] pos={p['position']} stars={p['stars_total']}"
            f" budget=R{p['rock_total']}/P{p['paper_total']}/S{p['scissors_total']}"
            f" | Alive opponents: {len(self._alive_opponents())}"
        )

    def close(self):
        pass
