# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## 0.4.0 (2023-09-13)

### Added
- Python 3.11 support ([#175](https://github.com/AI4S2S/s2spy/pull/175)).
- Support monthly and weekly data for preprocess module ([#173](https://github.com/AI4S2S/s2spy/pull/173)).

### Changed
- Example data used in the tutorial notebooks is now downloaded from Zenodo ([#172](https://github.com/AI4S2S/s2spy/pull/172)).

## 0.3.0 (2023-03-08)

### Added
- "Label alignment" functionality for RGDR, to align labels over multiple train-test splits ([#144](https://github.com/AI4S2S/s2spy/pull/144)).
- A preprocessing module, which can be used to calculate climatology/anomalies and to detrend data ([#152](https://github.com/AI4S2S/s2spy/pull/152)).
- Support for specifying multiple target and precursor intervals in RGDR ([#153](https://github.com/AI4S2S/s2spy/pull/153)).

### Changed
- A bug in the spherical area calculation of RGDR has been fixed ([#133](https://github.com/AI4S2S/s2spy/pull/133)).
- Default settings for RGDR have been removed. Users now need to fully specify their RGDR setup ([#133](https://github.com/AI4S2S/s2spy/pull/133)).
- The RGDR visualization plots are now called using `RGDR.preview_correlation` and `RGDR.preview_clusters` ([#106](https://github.com/AI4S2S/s2spy/pull/106)).

### Removed
- Calendar, resampling, and traintest modules have been moved to a separate package named [Lilio](https://github.com/AI4S2S/lilio) ([#158](https://github.com/AI4S2S/s2spy/pull/158)).

### Dev changes
- Use `hatch` as the project manager, and `ruff` as the linter ([#159](https://github.com/AI4S2S/s2spy/pull/159)).
- Notebooks have been moved to the docs folder, to be included in ReadtheDocs in the future ([#159](https://github.com/AI4S2S/s2spy/pull/159)).

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