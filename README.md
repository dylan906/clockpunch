# README

## Installation

- Recommend installing into clean Python environment with `$ conda create --name [my_env] python=3.8.5`
- To install, enter `$ pip install -e .`
- To install optional developer dependencies, enter `$ pip install -e .[dev]`
- The above steps seem to not work with installing Ray, so you will have to install Ray modules manually with:
  - `pip install ray[tune]`
  - `pip install ray[rllib]`
  - `pip install ray[air]`
 
## Troubleshooting

- If a test is mysteriously stopped while running, check if it is stuck at a `matplotlib` command. If so, launch *XLaunch* if it's not already running.
- To update the registration info (name, entry point) of the Gym environment, remember to update the register command in `scheduler_testbed/__init__.py`