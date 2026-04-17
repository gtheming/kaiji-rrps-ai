from gym_core.Q_learn import RRPSQLearnAgentCore
from environment_static.rrps_gym import RestrictedRPSEnv
from environment_static.rrps_gym import Observation
import numpy as np


class QLearnTabularNav(RRPSQLearnAgentCore):
    def __init__(self):
        super().__init__(env=RestrictedRPSEnv(n_opponents=1, stars=3))

    def hash(self, obs):
        ag = obs["agent"]
        opp = obs["opponent"]

        rel_dx = np.sign(opp["position"][0] - ag["position"][0])  # -1, 0, 1
        rel_dy = np.sign(opp["position"][1] - ag["position"][1])  # -1, 0, 1
        key = (
            ag["stars"],
            ag["budget"]["rock"],
            ag["budget"]["paper"],
            ag["budget"]["scissors"],
            opp["stars"],
            opp["budget"]["rock"] > 0,
            opp["budget"]["paper"] > 0,
            opp["budget"]["scissors"] > 0,
            rel_dx,
            rel_dy,
        )
        return key
