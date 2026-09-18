"""Task 1: your own custom Gymnasium environment.

Design the world yourself. The requirements it has to meet are in the assignment
readme.

Delete this docstring and describe your own world instead.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.spaces import Discrete


class MyEnv(gym.Env):
    """TODO: one line on what this world is and what the agent is trying to do."""
    """robot is trying to reach every tower with a limited amount of battery and with a non-guarenteed chance of moving correctly"""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    MAX_BATTERY = 5
    N_TOWERS = 7

    ADVANCED_PROBABILITY = .9
    DELAYED_PROBABILITY = .07
    PUSHBACK_PROBABILITY = .03

    EXPRESS_ADVANCED_PROBABILITY = .7
    EXPRESS_DELAYED_PROBABILITY = .2
    EXPRESS_PUSHBACK_PROBABILITY = .1

    def __init__(self, render_mode: str | None = None):
        """
            the world consists of 7 towers and a robot
            the robot has a battery of 5 (range 0-5)
            each move consumes 1 battery
            the robot at attempting to move has an 80% chance of moving forward, 15% chance of doing nothing, and 5% chance of going backwards
            there are two types of ways to move to a tower, express and normal lane
            express has a lower chance of success but moves 2 towers and takes 2 battery, normal moves 1 tower with a higher rate
        """

        self.observation_space = Discrete((self.MAX_BATTERY+1) * self.N_TOWERS)  # battery can also be at 0
        self.action_space = Discrete(3)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        # start at tower 0 with max battery
        self.tower_loc = 0
        self.battery = self.MAX_BATTERY

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not awork and
        # the reproducibility test fails.
        super().reset(seed=seed)

        self.tower_loc = 0
        self.battery = self.MAX_BATTERY

        return self._get_obs(), self._get_info()

    def _handleAdvance(self, success_rate, fail_rate, advance_amt, energy_usage):
        """
        handles advance attempts by the robot
        Args:
            success_rate: chance of advance happening
            fail_rate: chance that advance will fail
            advance_amt: the amt of spaces the robot will advance upon success occuring

        """
        rand_num = self.np_random.random()
        if rand_num < success_rate:  # chance of success
            self.tower_loc += advance_amt
        elif rand_num > success_rate + fail_rate:  # probability of going wrong way
            self.tower_loc -= 1
        # the else is that nothing happens

        self.battery -= energy_usage

        # prevent overflow
        if self.tower_loc >= self.N_TOWERS - 1:
            self.tower_loc = self.N_TOWERS - 1
        if self.tower_loc < 0:
            self.tower_loc = 0

    def step(self, action: int):
        # TODO: apply the action, with noise drawn from self.np_random.
        #
        # Return terminated=True when the episode genuinely ends -- goal reached,
        # agent died, game over. Leave truncated as False and let the TimeLimit
        # wrapper from register() handle running out of time. The agent treats
        # the two differently, and so should you.
        terminated = False
        truncated = False
        reward = 0

        if action == 0:  # charge
            self.battery += 1
            if self.battery > self.MAX_BATTERY:
                self.battery = self.MAX_BATTERY
        elif action == 1:  # advance
            if self.battery <= 0:  # advanced with no battery
                terminated = True
                reward -= 20
            else:
                self._handleAdvance(self.ADVANCED_PROBABILITY, self.PUSHBACK_PROBABILITY, 1, 1)
        elif action == 2:  # express advance
            if self.battery < 2:  # advanced with not enough battery
                terminated = True
                reward -= 20
            else:
                self._handleAdvance(self.EXPRESS_ADVANCED_PROBABILITY, self.EXPRESS_PUSHBACK_PROBABILITY, 2, 2)
        else:
            raise ValueError(f"Invalid action: {action}")

        if self.tower_loc == self.N_TOWERS - 1:
            terminated = True
            reward += 20
        else:
            reward -= 1

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode != "ansi":
            return None
        render_str = f"Current Tower: {self.tower_loc}\n"
        f"Battery: {self.battery} / {self.MAX_BATTERY}\n"

        for i in range(self.N_TOWERS):
            if i == self.tower_loc:
                render_str += "R "
            else:
                render_str += "T "
        render_str += "\n"

        return render_str

    def close(self):
        self.render_mode = None

    def _get_obs(self):
        return self.tower_loc * (self.MAX_BATTERY + 1) + self.battery  # simple hash function

    def _get_info(self):
        return {
            "tower": self.tower_loc,
            "battery": self.battery
        }


# TODO: name your environment. The id must start with "cs272/" and end with a
# version, and max_episode_steps must be large enough that a competent agent can
# finish but small enough that a lost one gives up.
register(
    id="cs272/ChargeTower-v0",
    entry_point="myenv:MyEnv",
    max_episode_steps=300,
)
