from gym_core.Q_learn import RRPSQLearnCore
from environment_tabular_nav.rps_gym import Observation
import numpy as np


class QLearnTabularNav(RRPSQLearnCore[Observation]):
    def hash(self, obs):
        ag = obs["agent"]
        opp = obs["opponent"]
        rel_dx = np.sign(opp["position"][0] - ag["position"][0])
        rel_dy = np.sign(opp["position"][1] - ag["position"][1])
        initial = self.env.initial_budget
        opp_state = self.env.player_dict[opp["player_id"]]
        return (
            ag["stars_total"],
            ag["budget"]["rock_total"] > 0,
            ag["budget"]["paper_total"] > 0,
            ag["budget"]["scissors_total"] > 0,
            opp["stars_total"],
            initial - opp_state["rock_total"],
            initial - opp_state["paper_total"],
            initial - opp_state["scissors_total"],
            rel_dx,
            rel_dy,
        )
