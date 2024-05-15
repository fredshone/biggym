[![Daily CI Build](https://github.com/fredshone/biggym/actions/workflows/daily-scheduled-ci.yml/badge.svg)](https://github.com/fredshone/biggym/actions/workflows/daily-scheduled-ci.yml)

# BIGGYM

Representing activity-based modeling problems as reinforcement learning problems using the [gymnasium](https://gymnasium.farama.org/) api.

## Individual Scheduling

- `biggym.envs.SchedulerEnv`: Simple "scheduling" challenge for a single agent with minimal interaction with world.
  - agent has three possible states: at home, at work, traveling
  - agent has two possible actions: travel to work, travel to home
  - reward is based on MATSim utility score

### Proposed extensions

- allow additional activity choices (participation)
- add stochastics to travel times
- add mode choice
- create continuous choices (_ie "go to work for N hours"_)
- inverse RL
- electric vehicle charging

## Household Scheduling


## Multi-Agent Scheduling


## Installation

To install we recommend using the [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager:

### As a developer
<!--- --8<-- [start:docs-install-dev] -->
``` shell
git clone git@github.com:arup-group/biggym.git
cd biggym
mamba create -n biggym --file requirements/base.txt --file requirements/dev.txt
mamba activate biggym
pip install --no-deps -e .
```
<!--- --8<-- [end:docs-install-dev] -->