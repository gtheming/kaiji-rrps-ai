from abc import ABC, abstractmethod, classmethod
from gym_core.info import Info
import gymnasium as gym
from gym_core.rrps_gym import RRPSEnvCore


class QLearn:
    def __init__(self, env: RRPSEnvCore) -> None:
        if not isinstance(env, RRPSEnvCore):
            raise TypeError(
                "env must be an instance of RRPSEnvCore, got"
                f" {type(env).__name__}"
            )
        self.env = env
    