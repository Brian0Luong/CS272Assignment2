"""Task 2: SARSA(lambda) with eligibility traces.

Do not change the class name or the constructor signature -- the grading harness
constructs this class directly, and it will hand you an environment you have
never seen. Read the sizes off the spaces and never assume anything about what a
state number means. Do not import myenv from this file.
"""

from typing import Any

import numpy as np
import gymnasium as gym

ACCUMULATING = "accumulating"
REPLACING = "replacing"


def argmax_action(values: np.ndarray, rng: np.random.Generator) -> int:
    """Return the index of the largest value, breaking ties uniformly at random."""
    best = np.flatnonzero(values == np.max(values))
    return int(rng.choice(best))


class SarsaLambdaAgent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        alpha: float = 0.05,
        eps: float = 0.1,
        lam: float = 0.9,
        trace: str = ACCUMULATING,
        total_epi: int = 5_000,
        init_val: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            env: any tabular gym environment. Both spaces are Discrete.
            gamma: discount factor.
            alpha: learning rate.
            eps: exploration rate for a plain (non-decaying) epsilon-greedy.
            lam: the lambda of SARSA(lambda), in [0, 1]. At 0 this must reduce
                to ordinary one-step SARSA.
            trace: "replacing" or "accumulating".
            total_epi: number of training episodes.
            init_val: value every q(s,a) starts at. Setting this at or slightly
                above the best achievable return makes every untried action look
                good, which drives systematic exploration -- on a sparse-reward
                environment that is often what makes learning possible at all.
            seed: seed for the agent's own randomness, for reproducible runs.
        """
        if trace not in (ACCUMULATING, REPLACING):
            raise ValueError(f"unknown trace type: {trace}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= eps <= 1.0:
            raise ValueError("eps must be in [0, 1]")
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")

        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.lam = lam
        self.trace = trace
        self.total_epi = total_epi
        self.init_val = init_val
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.q = self.init_qtable(init_val)

    def init_qtable(self, init_val: float = 0.0) -> np.ndarray:
        return np.full((self.n_states, self.n_actions), init_val, dtype=float)

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        if exploration and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        return argmax_action(self.q[state], self.rng)

    def learn(self) -> list[float]:
        returns: list[float] = []

        for episode_number in range(self.total_epi):
            reset_seed = self.seed if episode_number == 0 else None
            state, _ = self.env.reset(seed=reset_seed)
            action = self.eps_greedy(state)
            eligibility = np.zeros_like(self.q)
            total_return = 0.0

            while True:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_return += reward

                if terminated:
                    delta = reward - self.q[state, action]
                    next_action = None
                else:
                    next_action = self.eps_greedy(next_state)
                    delta = reward + self.gamma * self.q[next_state, next_action] - self.q[state, action]

                if self.trace == ACCUMULATING:
                    eligibility[state, action] += 1.0
                else:
                    eligibility[state, action] = 1.0

                self.q += self.alpha * delta * eligibility
                eligibility *= self.gamma * self.lam

                if terminated or truncated:
                    break
                state = next_state
                action = next_action

            returns.append(total_return)

        return returns

    def best_run(self, max_steps: int = 300) -> tuple[list[tuple[int, int, float]], bool]:
        state, _ = self.env.reset(seed=self.seed)
        episode: list[tuple[int, int, float]] = []

        for _ in range(max_steps):
            action = self.eps_greedy(state, exploration=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            if terminated or truncated:
                return episode, terminated
            state = next_state

        return episode, False

    def calc_return(self, episode: list[tuple[Any, Any, float]], discounted: bool = False) -> float:
        if not discounted:
            return float(sum(reward for _, _, reward in episode))
        return float(sum((self.gamma**time) * reward for time, (_, _, reward) in enumerate(episode)))


class RandomAgent(SarsaLambdaAgent):
    """The baseline your agent has to beat. Already written; do not change it."""

    def learn(self) -> list[float]:
        returns = []
        for _ in range(self.total_epi):
            self.env.reset()
            total = 0.0
            while True:
                action = int(self.rng.integers(self.n_actions))
                _, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
        return returns
