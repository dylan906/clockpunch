# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/).
This project adheres to [Semantic Versioning](http://semver.org/) in spirit, but in reality major backward-compatibility-breaking changes are made between MINOR versions.
PATCH versions generally don't break interfaces.
I'm trying to get better about that. 

## [unreleased] - 20YY-MM-DD

### Added
- Added MinMax wrapper that scales every entry in a gym.Dict observation space by sklearn.MinMaxScaler (#4).

### Changed

### Deprecated

### Fixed
- Fixed bug in `builtTuner()` that prevented environment from building in Ray 0.2.5 (#3).

### Removed
- Removed FilterCovElements wrapper, deprecated by SplitArrayObs and FilterObs (#5).

## [0.6.0] - 2023-07-06

### Added
- Split repo off from original repo, which had become hopelessly entangled with specific studies.

### Changed
- Changed `NormalizedMetric` to `GenricReward` to be more clear, less redundant (#1).

### Deprecated

### Fixed

### Removed
