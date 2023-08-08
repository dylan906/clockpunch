# README
This repo (***punchclock***) is for a Gym/Gymnasium environment that models generic space object tracking sensors.
Parts of the codebase mostly work, but everything is extremely fragile. 
Also there is no documentation yet, so you are on your own for figuring out everything. 
When I have time I'll add examples and documentation, but that may never happen.
Good luck, we're all counting on you.

## Installation

- Recommend installing into clean Python environment with `$ conda create --name [my_env] python=3.8.5`
- To install: `$ pip install -e .`
- To install optional developer dependencies: `$ pip install -e .[dev]`
 
## Troubleshooting

- If a test is mysteriously stopped while running, check if it is stuck at a `matplotlib` command. If so, launch *XLaunch* if it's not already running.
- To update the registration info (name, entry point) of the Gym environment, remember to update the register command in `punchclock/__init__.py`