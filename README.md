# README

## Installation

- Recommend installing into clean Python environment with `$ conda create --name [my_env] python=3.8.5`
- To install: `$ pip install -e .`
- To install optional developer dependencies: `$ pip install -e .[dev]`
 
## Troubleshooting

- If a test is mysteriously stopped while running, check if it is stuck at a `matplotlib` command. If so, launch *XLaunch* if it's not already running.
- To update the registration info (name, entry point) of the Gym environment, remember to update the register command in `punchclock/__init__.py`