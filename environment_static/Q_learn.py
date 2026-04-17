from gym_core.Q_learn import RRPSQLearnCore

from environment_static.rrps_gym import Observation


class QLearnStatic(RRPSQLearnCore[Observation]):

    def hash(self, obs):
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
