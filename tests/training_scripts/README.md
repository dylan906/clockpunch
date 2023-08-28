# README

## Introduction
The `punchclock/training_scripts` folder has scripts for training and Monte Carlo sims, but no corresponding test scripts in this folder (`tests/training_scripts`). The relevant folder structure is:

```
punchclock\
    ...
    \punchclock
        ...
        \training_scripts
            bash_script_mc.sh
            bash_script_training.sh
            run_mc.script.py
            run_tune_script.py
    \tests
        ...
        \training_scripts
            ...
            gen_training_config.py
            config_test.json
            README.md
```
## How to test
- The bash script `bash_script_training.sh` can be tested by running it line by line in a bash terminal.
- The python script `run_tune_script.py` can be tested by pasting the below line into a bash terminal (the same line is contained in `bash_script_training.sh`):
    - `python punchclock/training_scripts/run_tune_script.py tests/training_scripts/config_training_test.json`
    - Note: Running the above line will save files to `tests/training_scripts/test_results`
- Note: If you want to change test parameters but don't want to manually modify `config_test.json`, you can modify and run the script `gen_training_config.py` to generate new config files.
  - This is the recommended way to modify the config file.