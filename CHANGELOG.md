# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/).
This project adheres to [Semantic Versioning](http://semver.org/) in spirit, but in reality major backward-compatibility-breaking changes are made between MINOR versions.
PATCH versions generally don't break interfaces.
I'm trying to get better about that. 

## [unreleased] - 20YY-MM-DD

### Added
- New `InfoWrapper`s: `CombineInfoItems`, `TransformInfoWithNumpy` (#51).

### Changed
- Replaced `CopyObsItem` with more flexible `CopyObsInfoItem` wrapper (#68).

### Deprecated

### Fixed


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
