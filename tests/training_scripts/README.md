# README
Readme for `training_scripts/` tests
## Introduction
The `punchclock/training_scripts` folder has scripts for training and Monte Carlo sims, but no corresponding test scripts in this folder (`tests/training_scripts`). The relevant folder structure is:

```
punchclock
    ...
    /training_scripts
        run_mc.script.py
        run_tune_script.py
tests
    ...
    /training_scripts
        /mc_results
        /training_results_*
        config_mc_test.json
        config_training_test.json
        env_config.json
        gen_mc_config.py
        gen_training_config.py
        README.md
```
## How to test
- The tests are dependent on the environment config file `env_config.json`.
  - The env config file is generated when running `gen_mc_config.py`.
- The python script `run_tune_script.py` can be tested by pasting the below line into a bash terminal:
    - `python punchclock/training_scripts/run_tune_script.py tests/training_scripts/config_training_test.json`
    - Note: Running the above line will save files to `tests/training_scripts/training_results_*/`. A new file (replacing `*`) is made every run.
- If you want to change test parameters in `config_training_test.json` or `config_mc_test.json`, you can modify and run the scripts `gen_training_config.py` or `gen_mc_config.py`, respectively, to generate new config files.
  - This is the recommended way to modify the config file.