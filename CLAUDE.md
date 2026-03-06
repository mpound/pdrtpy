# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pdrtpy** (PhotoDissociation Region Toolbox - Python) is a scientific Python package for astrophysical analysis of photodissociation regions (PDRs). It helps astronomers determine physical parameters (density, radiation field) from far-infrared and millimeter/submillimeter spectral observations from telescopes like ALMA, SOFIA, JWST, Spitzer, and Herschel.

- **Language**: Python 3.10+
- **Build System**: Hatchling (pyproject.toml)
- **Key Dependencies**: astropy, numpy <2.0, scipy, matplotlib, lmfit, emcee
- **Testing**: pytest across Python 3.10, 3.11, 3.12, 3.13

## Development Commands

### Setup
```bash
# Install in editable mode with all dependencies
uv pip install -e ".[all]"

# Or just core dependencies
uv pip install -e .
```

### Testing
```bash
# Run all tests
uv run pytest -n auto

# Run tests with coverage
uv run pytest --cov-report=xml --cov-config=pyproject.toml --cov=pdrtpy --cov=tests

# Run a single test file
uv run pytest -n auto pdrtpy/test/test_measurement.py

# Run a single test function
uv run pytest pdrtpy/tool/test/test_h2excitation.py::test_h2_single_pixel
```

### Linting and Formatting
```bash
# Run ruff linter
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code with black (line length: 120)
uv run black .
```

### Documentation
```bash
# Serve docs locally (auto-rebuild on changes)
uv run sphinx-autobuild pdrtpy/docs/source pdrtpy/docs/build -b html

# Build static HTML docs
uv run sphinx-build pdrtpy/docs/source pdrtpy/docs/build -b html
```

## Core Architecture

The codebase follows a **layered, modular architecture** with clear separation between data models, analysis tools, and visualization. Understanding how these layers interact is critical for working with the code.

### Layered Design

```
Data Models (Foundation)
├── Measurement (pdrtpy/measurement.py)
├── ModelSet (pdrtpy/modelset.py)
├── Molecule (pdrtpy/molecule.py)
└── pdrutils (pdrtpy/pdrutils.py)

Analysis Tools (Business Logic)
├── ToolBase (pdrtpy/tool/toolbase.py)
├── LineRatioFit (pdrtpy/tool/lineratiofit.py)
└── ExcitationFit classes (pdrtpy/tool/excitation.py)
    ├── H2ExcitationFit
    ├── COExcitationFit
    └── C13OExcitationFit

Visualization Layer
├── PlotBase (pdrtpy/plot/plotbase.py)
├── LineRatioPlot (pdrtpy/plot/lineratioplot.py)
├── ExcitationPlot (pdrtpy/plot/excitationplot.py)
└── ModelPlot (pdrtpy/plot/modelplot.py)
```

### Key Components

#### Measurement (pdrtpy/measurement.py)
- Wraps `astropy.nddata.CCDData` for spectral line/continuum observations
- Handles FITS I/O, uncertainty propagation, and beam parameters
- Supports both single-pixel values and spatial maps
- Mathematical operations (`/`, `*`) automatically propagate errors via `StdDevUncertainty`

#### ModelSet (pdrtpy/modelset.py)
- Manages pre-computed PDR model sets (Wolfire-Kaufman, Kosma-Tau variants)
- Models are stored as FITS files in `pdrtpy/models/` with hierarchical directory structure
- Metadata stored in astropy Tables for efficient querying
- Supports multiple metallicities, geometries, and viewing angles

#### Tool Classes (pdrtpy/tool/)
- **LineRatioFit**: Core fitting engine using lmfit minimization. Accepts measurements, creates intensity ratios, performs χ² minimization against models to determine density and radiation field.
- **ExcitationFit**: Complex fitting tool for molecular excitation diagrams. Supports 1-2 component fits with optional MCMC (via emcee).
- All tools inherit from `ToolBase` which provides measurement handling and property detection (maps vs. scalars vs. vectors).

#### Plot Classes (pdrtpy/plot/)
- **PlotBase**: Abstract base providing common plotting infrastructure, color normalization, contour generation
- **LineRatioPlot**: Visualizes fitting results from LineRatioFit with phase-space diagrams, model overlays
- **ExcitationPlot**: Excitation diagrams with interactive features (click pixels in maps)
- **ModelPlot**: Standalone model explorer (doesn't require fitting first)

## Important Architectural Patterns

### 1. Composition Over Inheritance
- Fitting tools accept `ModelSet` + `Measurements` as constructor arguments
- Plot classes reference parent Tool or ModelSet
- No tight coupling between fitting and plotting layers

### 2. Custom Astropy Units
The package defines custom units for radiation field strength (defined in `pdrutils.py`):
- **Habing**: G₀ (1.6e-3 erg cm⁻² s⁻¹)
- **Draine**: χ (radiation field strength)
- **Mathis**: Alternative radiation field unit

These are automatically registered on import. When working with radiation field values, always use these custom units.

### 3. Automatic Error Propagation
- `Measurement` leverages astropy's `CCDData` uncertainty system
- Arithmetic operations automatically propagate errors
- All uncertainties stored as `StdDevUncertainty`
- Critical for scientific accuracy in derived quantities

### 4. Spatial Data Support
- Tools automatically detect data dimensionality (scalar, vector, map)
- Same interface works for single-pixel and spatial maps
- Iterative pixel-by-pixel fitting for maps (results wrapped in `FitMap`)
- When adding new features, ensure they handle all three data types

### 5. Model Organization
- Models are pre-computed FITS files in hierarchical directory structure (`pdrtpy/models/`)
- Metadata in `all_models.tab` and model-specific tables
- scipy interpolation used for regridding models to match observations
- Users can add custom models via external `modelsetinfo` files

### 6. Molecular Data
- Molecular partition functions and transition data in `pdrtpy/tables/`
- Abstract `BaseMolecule` class with concrete implementations (H2, CO, C13O)
- Each molecule knows its own partition function and transition probabilities

## Code Style and Conventions

- **Formatter**: black with 120 character line length
- **Linter**: ruff (checks F, E, W, B, I, NPY, PD, RUF rules)
- **Import sorting**: isort with black profile
- **Docstrings**: NumPy docstring style
- **Pre-commit hooks**: Configured in `.pre-commit-config.yaml` (trailing whitespace, black, isort, ruff)

## Testing Notes

- Tests located in `pdrtpy/test/`, `pdrtpy/tool/test/`, `pdrtpy/plot/test/`
- Test data (FITS files) in `pdrtpy/testdata/`
- Tests run across Python 3.10, 3.11, 3.12, 3.13 in CI (Ubuntu, macOS, Windows)
- When adding new features, follow existing test patterns in similar modules
- Use pytest fixtures for common setup (see existing test files)
- Always use 'pytest -n auto' to run across all available CPU cores.

## Important Implementation Details

- **No external database**: All model data packaged as FITS files in the distribution
- **Interactive plotting**: Uses `mpl-interactions` for Jupyter exploration
- **MCMC integration**: `emcee` library used for Bayesian parameter estimation in excitation fits
- **Flexible I/O**: Measurements can come from FITS files or numpy arrays
- **Physics constants**: Centralized in `pdrutils.py` (Boltzmann constant, partition functions, etc.)
- **Prefer simple solutions**
- **Always work in a branch.** Never directly change the master branch.
