# Charge Tower Environment

General information about the environment:

| | |
| ------ | ------- |
| Action Space | `Discrete(3)` |
| Observation Space | `Discrete(42)` |
| Register Environment ID | `cs272/ChargeTower-v0` |
| Maximum Episode Steps | `300` |

## Description

The environment consists of 7 towers.
The drone starts with 5 units of battery and at tower 0.
Each action that does not terminate the episode gives a reward of -1. Reaching the final beacon gives a reward of 20, while attempting to advance without sufficient battery gives a reward of -20.
The drone can take one of two paths from any given tower, an express path and a normal path. 
On an attempt to advance from a given path, the drone can successfully advance, fail to advance, or be pushed back one tower with differing probabilities. 
The express path consumes 2 batteries and has a lower level of success, but advances 2 towers, while the normal path consumes 1 battery, but advances 1 tower with a higher rate of success.
The drone can also choose to charge, which will restore 1 battery and stay in the same location.
## Action Space

The action space is:

Discrete(3)

The available actions are: 

- 0: Charge

- 1: Advance

- 2: Express Advance


## Observation Space

The observation is a single integer representing the drone's current tower and battery level.

The state is represented by:
`tower_loc * (MAX_BATTERY + 1) + battery` returned as an integer.

For example tower location 2 with a battery of 4 would be
2 * 6 + 4 = 16. The number of possible observations is 42.

## Rewards

The reward structure is:

| Event                                       |   Reward |
| ------------------------------------------- | -------: |
| Taking am action without reaching the goal  |       -1 |
| Reaching the final beacon                   |      +20 |
| Attempting to advance with an empty battery |      -20 |
| Attempting to advance with less than 2 battery |      -20 |

## Starting State

- The drone is at tower `0`. 
- The drone has `5` units of battery.

## Transition Dynamics

When the agent selects a path, one of three outcomes occurs:

1. **Success:** The drone advances along the selected path.
2. **Failure:** The drone does not advance.
3. **Pushback:** The drone is pushed back by one tower.

The probabilities depend on the selected path.

### Normal Path

| Outcome  |     Probability (%) | Result                  |
| -------- | --------------: | ----------------------- |
| Success  | `90` | Advance 1 tower         |
| Failure  | `7` | Remain at current tower |
| Pushback | `3` | Move back 1 tower       |

The normal path consumes 1 unit of battery when attempted.

### Express Path

| Outcome  |     Probability (%) | Result                  |
| -------- | --------------: | ----------------------- |
| Success  | `70` | Advance 2 towers        |
| Failure  | `20` | Remain at current tower |
| Pushback | `10` | Move back 1 tower       |

The express path consumes 2 units of battery when attempted.
The transition outcome is randomly selected according to these probabilities on each action.
Batteries are consumed regardless of outcome.

The drone cannot move below tower 0 or beyond tower 6. Any movement that would go outside these bounds is clamped to the nearest valid tower.
## Termination

An episode terminates when the following happen.

```text
- The drone reaches the final beacon at `N_TOWERS - 1` (tower 6).
- The drone attempts a normal advance with 0 battery.
- The drone attempts an express advance with less than 2 battery.
```

Upon reaching the final beacon, the agent receives a reward of 20.

## Truncation
An episode is truncated after 300 steps if it has not already terminated.

Truncation is handled by the `TimeLimit` wrapper with a `max_episode_steps = 300`

## Arguments

`ChargeTower` accepts the following argument:

- `render_mode`: `str | None`, default `None`.
    - Set to `"ansi"` to enable text rendering.
    - If `None`, rendering is disabled.

The environment can be created with:

```python
import gymnasium as gym

env = gym.make(
    "cs272/ChargeTower-v0",
    render_mode="ansi"
)
```
- render_mode string used to turn on or off rendering. 
  * if render mode is set to `"ansi"` render will return the following format:
    
    ```text
    Current Tower: 2
    Battery: 3 / 5
    T T R T T T T
    ```

