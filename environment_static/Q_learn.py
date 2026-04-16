from gym_core.Q_learn import RRPSQLearnCore
from environment_static.rrps_gym import StaticRRPSEnv
from environment_static.rrps_gym import Observation
import gym_core.visualizer as vis


class QLearnStatic(RRPSQLearnCore):

    def hash(self, obs: Observation):
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

    def render_gui(self, terminated, truncated, info):
        if not vis.is_initialized():
            vis.init()
        vis.refresh(terminated, truncated, info)
