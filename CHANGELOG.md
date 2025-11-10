# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Added

- New ode module to calculate and integrate first, second and third order equations for growth of perturbations.
- New odeint module that implements RKQS solver, for numerical compatibility with C code.

## [0.2.1] - 2025-11-04

### Fixed

- Bump version number in pyproject.toml

## [0.2.0] - 2025-11-04

### Added

- Plot module with convenience plotting functions, demonstrated in the examples notebook.
- Markdown documentation of benchmark results and k-function nomenclature.

### Changed

- Precompute tridiagonal matrix factors needed for 2nd deriv calculations
- Move 2nd deriv calculation into main compute kernel

## [0.1.0] - 2025-11-02

Initial release.
