[![Daily CI Build](https://github.com/fredshone/biggym/actions/workflows/daily-scheduled-ci.yml/badge.svg)](https://github.com/fredshone/biggym/actions/workflows/daily-scheduled-ci.yml)

# BIGGYM

Representing activity-based modeling problems as reinforcement learning problems using the [gymnasium](https://gymnasium.farama.org/) api.

## Individual Scheduling

- `biggym.envs.SchedulerEnv`: Simple "scheduling" challenge for a single agent with minimal interaction with world.
  - agent has three possible states: at home, at work, traveling
  - agent has two possible actions: travel to work, travel to home
  - reward is based on MATSim utility score

- `biggym.envs.SchedulerModeEnv`: Extends above to include an additional activity "shop" and travel mode choice between car, bus and walk.
  - agent has 6 possible states: at home, work, shop and traveling by car, bus or walking
  - agent has 9 possible actions, travel to home, work or shop, each by either car, bus or walk
  - reward is based on MATSim utility score

### Outstanding issues

- how to deal with noops
- how to deal with multi-dim actions, eg participation and mode

### Proposed extensions

- allow additional activity choices (participation, location, time, mode, toll routing, charging, routing)
- separate choice dimensions better (eg participation and mode as different actions spaces)
- add stochastics to travel times
- multi-day
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