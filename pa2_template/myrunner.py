"""Reproducible checks, lambda sweep, plots, table, and sample episode."""

from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.utils.env_checker import check_env

import myenv
from myagent import RandomAgent, SarsaLambdaAgent

LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [11, 22, 33, 44, 55]
ENV_ID = "cs272/ChargeTower-v0"
ACTION_NAMES = ("CHARGE", "ADVANCE", "EXPRESS")
EPISODES = 5_000
SMOOTH_WINDOW = 100
TARGET_RETURN = 8.0
FINAL_WINDOW = 500


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average; early entries lack a complete window."""
    result = np.full_like(values, np.nan, dtype=float)
    result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def train_once(lam: float, seed: int) -> np.ndarray:
    env = gym.make(ENV_ID)
    agent = SarsaLambdaAgent(
        env, gamma=0.99, alpha=0.05, eps=0.10, lam=lam,
        trace="accumulating", total_epi=EPISODES, init_val=1.0, seed=seed,
    )
    returns = np.asarray(agent.learn(), dtype=float)
    env.close()
    return returns


def first_target_episode(smoothed: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(smoothed >= threshold)
    return int(hits[0] + 1) if hits.size else None


def save_map(path: Path) -> None:
    """Draw normal (+1 tower) and express (+2 towers) routes."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    xs = np.arange(7)

    for tower in range(6):
        ax.annotate(
            "", xy=(tower + 0.85, 0), xytext=(tower + 0.15, 0),
            arrowprops={"arrowstyle": "->", "color": "#23815b", "lw": 2},
        )

    for tower in range(5):
        destination = min(tower + 2, 6)
        ax.annotate(
            "", xy=(destination, 0.12), xytext=(tower, 0.12),
            arrowprops={
                "arrowstyle": "->", "color": "#d87932", "lw": 1.6,
                "connectionstyle": "arc3,rad=-0.35",
            },
        )

    ax.scatter(xs, np.zeros(7), s=650, color="#d9e8ff", edgecolor="#17365d", zorder=3)
    for x in xs:
        ax.text(x, 0, str(x), ha="center", va="center", fontsize=9, weight="bold")
    ax.text(0, -0.19, "START", ha="center", fontsize=9)
    ax.text(6, -0.19, "BEACON", ha="center", fontsize=9)
    ax.text(3, 1.02, "Orange: EXPRESS skips one tower (costs 2 battery)", ha="center")
    ax.text(3, -0.38, "Green: ADVANCE moves one tower (costs 1 battery)", ha="center")
    ax.text(3, -0.68, "CHARGE at every tower; battery 0–5 gives 42 states. From 5, both routes lead to 6.", ha="center")
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(-0.88, 1.22)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    check_env(myenv.MyEnv(render_mode="ansi"), skip_render_check=False)

    rows = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for lam in LAMBDAS:
        raw = np.vstack([train_once(lam, seed) for seed in SEEDS])
        smooth = np.vstack([moving_average(run, SMOOTH_WINDOW) for run in raw])
        mean = smooth.mean(axis=0)
        std = smooth.std(axis=0)
        x = np.arange(1, EPISODES + 1)
        ax.plot(x, mean, label=f"λ={lam:g}")
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)

        target_episodes = [first_target_episode(run, TARGET_RETURN) for run in smooth]
        reached = [value for value in target_episodes if value is not None]
        rows.append({
            "lambda": lam,
            "mean_episodes_to_target": round(float(np.mean(reached)), 1) if reached else "not reached",
            "seeds_reaching_target": f"{len(reached)}/{len(SEEDS)}",
            "mean_final_return": round(float(raw[:, -FINAL_WINDOW:].mean()), 3),
            "final_return_std_across_seeds": round(float(raw[:, -FINAL_WINDOW:].mean(axis=1).std()), 3),
        })

    ax.axhline(TARGET_RETURN, color="black", linestyle="--", linewidth=1, label="target return")
    ax.set(title="Charge Tower SARSA(λ) Learning Curves", xlabel="Training episode", ylabel="100-episode moving-average return")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "lambda_sweep.png", dpi=180)
    plt.close(fig)

    with (output_dir / "lambda_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    random_env = gym.make(ENV_ID)
    random_env.reset(seed=SEEDS[0])
    random_returns = RandomAgent(random_env, total_epi=EPISODES, seed=SEEDS[0]).learn()
    random_env.close()
    with (output_dir / "baseline.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Random mean return over {EPISODES} episodes: {np.mean(random_returns):.3f}\n")
        handle.write(f"Random final-{FINAL_WINDOW} mean: {np.mean(random_returns[-FINAL_WINDOW:]):.3f}\n")

    best_lambda = max(rows, key=lambda row: float(row["mean_final_return"]))["lambda"]
    demo_env = gym.make(ENV_ID, render_mode="ansi")
    demo_agent = SarsaLambdaAgent(demo_env, lam=float(best_lambda), total_epi=EPISODES, seed=2026)
    demo_agent.learn()
    state, _ = demo_env.reset(seed=272)
    transcript = [demo_env.render()]
    total = 0.0
    for step in range(1, demo_env.spec.max_episode_steps + 1):
        action = demo_agent.eps_greedy(state, exploration=False)
        state, reward, terminated, truncated, _ = demo_env.step(action)
        total += reward
        transcript.append(f"\nStep {step}: action={ACTION_NAMES[action]}, reward={reward:.1f}\n{demo_env.render()}")
        if terminated or truncated:
            break
    transcript.append(f"\nEpisode return: {total:.1f}; terminated={terminated}; truncated={truncated}")
    (output_dir / "sample_episode.txt").write_text("\n".join(transcript), encoding="utf-8")
    demo_env.close()

    save_map(output_dir / "environment_map.png")
    print("Environment check passed. Results saved in results/.")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
