# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Dev changes
- Use `hatch` as the project manager.
- Use `ruff` as the linter.
- Notebooks have been moved to the docs folder, to be included in ReadtheDocs in the future.

### Removed
- Calendar/resample and traintest modules have been moved to [Lilio](https://github.com/AI4S2S/lilio).

## 0.2.1 (2022-09-02)

### Fixed
- Display of images on ReadtheDocs and PyPi ([#97](https://github.com/AI4S2S/s2spy/pull/97))

## 0.2.0 (2022-09-01)

### Added
- Improve Sphinx documentation hosted on ReadtheDocs ([#32](https://github.com/AI4S2S/s2spy/pull/32) [#70](https://github.com/AI4S2S/s2spy/pull/70))
- Support max lags and mark target period methods in time module ([#40](https://github.com/AI4S2S/s2spy/pull/40) [#43](https://github.com/AI4S2S/s2spy/pull/43))
- Add traintest splitting module for cross-validation ([#37](https://github.com/AI4S2S/s2spy/pull/37))
  - Support sklearn splitters for traintest module ([#53](https://github.com/AI4S2S/s2spy/pull/53))
  - Implement train/test splits iterator ([#70](https://github.com/AI4S2S/s2spy/pull/70))
- Add Response Guided Dimensionality Reduction (RGDR) module ([#68](https://github.com/AI4S2S/s2spy/pull/68))
  - Implement correlation map function ([#49](https://github.com/AI4S2S/s2spy/pull/49))
  - Implement dbscan for RGDR ([#57](https://github.com/AI4S2S/s2spy/pull/57))
  - Support for multiple lags in RGDR ([#85](https://github.com/AI4S2S/s2spy/pull/85))
- Update Readme ([#95](https://github.com/AI4S2S/s2spy/pull/95))

### Changed
- Refactor resample methods as functions ([#50](https://github.com/AI4S2S/s2spy/issues/50))
- Refactor calendars to BaseCalendar class and subclasses ([#60](https://github.com/AI4S2S/s2spy/pull/60))

### Removed
- Python 3.7 support ([#65](https://github.com/AI4S2S/s2spy/issues/65))

## 0.1.0 (2022-06-28)

### Added
- Time module for an "advent calendar" to handle target and precursor periods.
- Implemented resampling data to the advent calendar.
- Example notebooks on how to use the calendar and resampling functionalities.

[Unreleased]: https://github.com/AI4S2S/s2spy