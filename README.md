# Hydrogen Embrittlement in Inconel 600 — CZM Framework with ML Analysis

Simulation, post-processing, and machine-learning scripts developed for
an MEng final-year dissertation at the University of Manchester (School
of Engineering, 2025–26).

## Overview

A two-dimensional polycrystalline cohesive zone model of Inconel 600 is
coupled to a uniform pre-charged hydrogen field via the Langmuir–McLean
isotherm and the Serebrinsky polynomial for hydrogen-coverage-dependent
cohesive strength. A seven-point parametric sweep over initial hydrogen
concentration is processed into a small dataset and used to train a
Random Forest surrogate, interpreted through TreeExplainer SHAP
attributions. Full methodological detail is given in the dissertation
report.

## Structure

- `simulation/` — Abaqus input-deck generators, USDFLD subroutine, and
  SLURM submission scripts for the parametric sweep.
- `postprocessing/` — ODB extraction (UTS, fracture strain) and
  Random Forest surrogate training with SHAP analysis.
- `mesh/` — documentation for regenerating the Neper-meshed polycrystal.

## Dependencies

- Abaqus 2024 with Fortran user-subroutine support (Intel 17.0.7)
- Neper 4.10.1
- Python 3.x with scikit-learn, pandas, numpy, matplotlib, shap
- Python 2.7 within the Abaqus runtime (for ODB API in
  `extract_results.py`)

## Reproducing the parametric sweep

1. Regenerate the mesh (see `mesh/README.md`).
2. Build the seven input decks: `python simulation/build_inp_static_v3.py`.
3. Submit on a SLURM cluster: `bash simulation/launch_sweep.sh`.
4. Extract results: `abaqus python postprocessing/extract_results.py`.
5. Train surrogate: `python postprocessing/analysis.py`.

## Citation

If you use this code, please cite the accompanying dissertation:

> Abu Joudeh, S. (2026). *Modelling Hydrogen Embrittlement in Inconel
> 600 Using a Cohesive Zone Framework with Machine Learning Analysis*.
> MEng dissertation, The University of Manchester.

## License

MIT. See `LICENSE`.
