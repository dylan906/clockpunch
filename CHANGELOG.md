# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).
This project adheres to [Semantic Versioning](http://semver.org/) in spirit, but in reality major backward-compatibility-breaking changes are made between MINOR versions.
PATCH versions generally don't break interfaces.
I'm trying to get better about that.

## [unreleased] - 20YY-MM-DD

### Added

### Changed
- `TransformInfoWithNumpy` now works if `key` is not provided by creating a new entry `info[new_key]`.
Allows using `np.ones` and other functions that don't need dynamic args (#121).

### Deprecated

### Fixed
- Fixed bug in building Ray policies using algorithm checkpoints.

### Removed

## [v0.8.4] - 2024-04-12

### Added

### Changed
- Replaced `single_sim_mode` in `MonteCarloRunner` with separate `save_last_step_only` and `static_initial_conditions` args (#118).
- `CustomCallbacks` no longer assign None to custom metrics that don't exist, instead do not create the those metrics (#119).

### Deprecated

### Fixed
- Fixed bug in `SimRunner` that would cause recurrent nets to error when getting a dict observation (#120).

### Removed

## [0.8.3] - 2024-01-29

### Added
- Added feature to `NumWindows` wrapper to return time until next window (#102, #111).
- Added `VisMap(InfoWrapper)` wrapper to calculate visibility matrix (outside of base env). Allows use of continuous visibility function value and its time derivative (#105, #106, #111).
- New module `visibility.py` that has low-level visibility functions (#107).
- Added `orbits.py` which has a collection of equations to calculate orbital parameters. Consolidated some equations into new module (#107).
- Added `TimeDiff` wrapper which takes time difference (and optional discrete derivative) of an item in environment info (#108).
- Added `CheckNanInf` wrapper to check if NaNs or Infs exist in observations or reward (#110).
- Added `ConvertDictItemSpaceDtype` wrapper to convert the dtype of an item in the observation space (#112).
- Added `ModifyNestedDict` wrapper to append/delete an item from a nested observation space or info (#113).
- Added constants for geostationary orbit radius/altitude (#116).

### Changed
- `ConfigurableCurriculumFn` now doesn't allow for level skipping, always starts curriculum at level 0, and repeats final level if env is already at final level (#101).
- `TransformDictObsWithNumpy` now more similar in structure and functionality to `TransformInfoWithNumpy` (#109).
- Patience is now optional in `SequentialCurriculumFn` (#115).

### Deprecated

### Fixed
- Fixed bug in `ecef2eci`/`eci2ecef` conversions that caused incorrect values (#117).

### Removed

## [0.8.2] - 20YY-11-08

### Added
- ezUKF now has option for random Q, R, p_init (#86).
- Added KL Divergence wrapper (#94).
- Added entropy difference wrapper (#95).
- Curriculum learning via the new module `currivulum.py` (#96).

### Changed
- Experiment names now have a random string appended to manual name (#89).
- `SSASchedulerParams` now has defaults for all args except horizon (#97).
- `MonteCarloRunner` now allows for multiple env configs to be passed in (#99).

### Deprecated

### Fixed
- Old restore Tuner function resurrected and fixed (#91).

### Removed

## [0.8.1] - 2023-10-18

### Added

- New `InfoWrapper`s: `CombineInfoItems`, `TransformInfoWithNumpy` (#51), `GetNonZeroElements` (#80), `ConfigurableLogicGate` (#84), `InfoFilter` (#73).
- New miscellaneous wrappers: `OperatorWrapper` (#69, #80), `MaskViolationChecker` (#75), `TruncateIfNoCustody` (#85).
- New `ObservationWrapper`: `MakeObsSpaceMultiBinary` (#74).
- `NumWindows` wrapper can now be configured to run in open-loop mode (#76).
- New model which layers N-number of FC layers with a single LSTM layer (#81, #82, #83).

### Changed

- Replaced `CopyObsItem` with more flexible `CopyObsInfoItem` wrapper (#68).
- `addPostProcessedCols` calculated fewer very specific use case metrics. Retained the most useful metrics (e.g. uncertainty measures). Moved the specific use case functions to separate module (#52).
- Changed the base class of the following wrappers from `ObservationWrapper` to `ModifyObsOrInfo`: `CustodyWrapper` (#71),`ConvertCustody2ActionMask` (#72), `VisMap2ActionMask` (#77) (#78).

### Deprecated

### Fixed

- Fixed edge case bug where `InfoWrapper` would incorrectly pass in action=None to `self.updateInfo()` (#70).
- Fixed various bugs and clarified type hints/doc strings in some analysis util functions (#52).
- `SimRunner` no longer fails to fetch info with some wrappers (#63).

### Removed

## [0.8.0] - 2023-09-27

### Added

- Sim results now include action mask violations (#53).
- Added function to generate random agents (used for debugging/tests) (#55).
- Wrapper that copies item from info to observation (#59).
- New `InfoWrapper` base class and module for info wrappers (#60).
- New wrapper `NumWindows(InfoWrapper)` dynamically updates number of sensor-target access windows left (#38).
- Added reverse-nest option to `NestObsItems` wrapper (#65).
- Added `LogisticTransformInfo` wrapper (#51).

### Changed

- Made null actions optionally ignorable in `MaskReward` wrapper (#48).
- Some tests for `SimRunner` now make more sense given the increased emphasis on wrappers (#50).
- Changed behavior of `RewardBase` to sum unwrapped reward with new (wrapped) reward (#48).
- Cleaned up `test/analysis_utils` (#54).
- `RewardBase` no longer sums unwrapped with wrapped reward (#66).
- Changed the following wrappers from `RewardWrapper` to `InfoWrapper`: `MaskReward`->`MaskViolationCounter`, `NullActionReward`->`ActionTypeCounter`, `ThresholdReward`->`ThresholdInfo` (#51).

### Deprecated

### Fixed

- `SimRunner._getObs()` is cleaner and now works with wrappers that don't have an `observation()` method (#49, #50).
- `ezUKF` now accepts optional initialization time as argument (54672fe).
- Fixed bug where `buildSpace` could create noncompliant `MultiDiscrete` spaces (#58).
- `buildPolicy` now works with Inf args (#38).

### Removed

- Removed old `ActionMask` wrapper in favor of more modular wrappers.
- Removed number of access windows left tracking from base environment (#62).
- Removed reward function settings from base environment and associated reward function class (#27).

## [0.7.0] - 2023-09-06

### Added

- Added wrapper to convert Box space to MultiBinary (#21).
- Added wrapper to mask wasted actions (#25).
- Added multiple wrappers to eventually replace the custom reward function scheme with a less-custom, more flexible version based on wrappers (#26).
- Added `LogisticTransformReward` wrapper (#32).
- New custom policy `MultiGreedy` that generically applies egreedy to arrays column-wise (#37).
- Add wrapper to transform scaler observations in a Dict obs space (#45).

### Changed

- Changed base env observation space. Full covariance matrices now included (vice just diagonals) (#15).
- `VisMap2ActionMask` and `ConvertCustody2ActionMask` now return 2d action masks (#18).
- Wrapper map in `build_env.py` replaced with automatic system. No longer need to add new wrappers to a variable another function (#29).
- Custom policies now use 2d action masks, consistent with wrappers (#37).
- `MultiplyObsItems` works with >2 spaces (#39).
- Updated to Python 3.10.0.
- `SimRunner` now relies on `IdentityWrapper` to determine when to split off custom policy observation space from Ray policy observation space (replaced static checking for observation space entries in a dict) (#46).

### Deprecated

### Fixed

- `/training_scripts` reorganized and file names made consistent with each other (#30).
- Custom policies and simulation runner now accept envs with both `Box` and `MultiBinary` action masks (#31).
- `MinMaxScaleDictObs` wrapper now doesn't convert arrays of 1s to 0s (#33).
- Fixed bug where `MultiBinaryConfig.fromSpace()` would error on 1d spaces (#34).
- Fixed dtype bug in `MinMaxScaleDictObs` (#35).
- Base env now does not update measurements if estimated non-visible target is tasked (#36).
- `MultiplyObsItems` works with non-float and mixed dtypes now (#39).

### Removed

- Removed old (non-functional) UCB policy (#37).

## [0.6.1] - 2023-08-08

### Added

- Added MinMax wrapper that scales every entry in a gym.Dict observation space by sklearn.MinMaxScaler (#4).
- Added some wrappers to help with array dimension handling and scaling (#5).
- Added a wrappers to track target custody and convert custody array to action mask (#10, #17).
- Added a wrapper to convert 2D covariance matrices to 3D (#16).

### Changed

- Changed a couple of low-level attribute names that shadowed `builtin` and appeared throughout repo (#9).
- Refactored `IntersectMask` wrapper to more generic `MultiplyObsItems` (#14).
- `ActionMask` wrapper deprecated in favor of combination of modular wrappers that can be used in other use cases as well (#13).
- saveJSONFile now works if you include ".json" in the file name.

### Deprecated

### Fixed

- Fixed bug in `builtTuner()` that prevented environment from building in Ray 0.2.5 (#3).
- Properly enforced order of `Dict` observation spaces in all relevant wrappers (#7).
- MC tests (`mc_config.py` and `mc.py`) are no longer reliant on an old copy/pasted config file (#24).

### Removed

- Removed FilterCovElements wrapper, deprecated by SplitArrayObs and FilterObs (#5).
- Removed old references to `need_obs` policy (#8).

## [0.6.0] - 2023-07-06

### Added

- Split repo off from original repo, which had become hopelessly entangled with specific studies.

### Changed

- Changed `NormalizedMetric` to `GenricReward` to be more clear, less redundant (#1).

### Deprecated

### Fixed

### Removed
