# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Full documentation site: installation, quickstart, a per-stage user guide,
  and a per-module API reference.

### Fixed

- `mkdocstrings` now parses the codebase's NumPy-style docstrings. Parameter,
  return, and attribute sections previously rendered as unstyled prose.
- Mermaid diagrams now render as diagrams rather than inert code blocks.
- Corrected several documentation examples that did not match the code:
  `despike_dataframe` and `InstrumentConfig` import paths, the
  `plot_cospectra` argument name and its `(fig, axes)` return, the
  `enrich_results_with_means` signature, and the argument order of
  `tf_sonic_line_averaging`.

### Changed

- Google-style docstring sections in `config`, `results`, and `data_quality`
  converted to NumPy style to match the rest of the codebase.

## 0.2.2

### Changed

- `__version__` is derived from the installed distribution metadata rather
  than being hard-coded.

## 0.2.1

### Added

- Package published to PyPI as `taylorswift-spectra`. The import name remains
  `TaylorSwift`.

## 0.2.0

### Added

- Vickers & Mahrt (1997) raw-data screening (`screening` module) with
  `vm97_*` diagnostic flags on every `SpectralResult`.
- Spectral transfer functions split into a dedicated `transfer_functions`
  module.

### Changed

- `data_quality` refactored; `cospectra` became a backward-compatible
  re-export hub.

---

!!! note
    Releases before 0.2.0 predate this changelog. See the
    [commit history](https://github.com/inkenbrandt/TaylorSwift/commits/main)
    for details.
