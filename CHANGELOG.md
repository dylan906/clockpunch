# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/).
This project adheres to [Semantic Versioning](http://semver.org/) in spirit, but in reality major backward-compatibility-breaking changes are made between MINOR versions.
PATCH versions generally don't break interfaces.
I'm trying to get better about that. 

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

### Deprecated

### Fixed
- Fixed bug in `builtTuner()` that prevented environment from building in Ray 0.2.5 (#3).
- Properly enforced order of `Dict` observation spaces in all relevant wrappers (#7).

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
