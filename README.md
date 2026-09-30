# OptiCut: Parallel Shape Optimization with CutFEM in FEniCSx

[![CI Build](https://github.com/Amina-ELB/OptiCut/actions/workflows/tests.yml/badge.svg)](https://github.com/Amina-ELB/OptiCut/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://amina-elb.github.io/OptiCut/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="./doc/images/opticut-logo-v2.png" alt="OptiCut Logo">
</p>

## Overview

**OptiCut** is an open-source research code dedicated to shape optimization using immersed boundary methods CutFEMx. It is implemented on top of the **FEniCSx** computing platform and **CutFEMx**.

The structural boundary is represented implicitly via the **Level Set method**, which governs its geometric evolution through a transport equation. A central contribution of this code is the integration of the **Cut Finite Element Method (CutFEM)**, via the `cutfemx` library, as the underlying PDE solver. This approach yields highly accurate approximations of the mechanical fields in the immediate vicinity of the structural boundary, a key advantage over classical Ersatz material methods, without requiring any mesh conforming or remeshing.

OptiCut supports distributed parallel computing (MPI) and is capable of handling fully 3D structural cases.

## Key Features

- **Immersed Boundary Methods:** Accurate mechanical evaluation at the boundary using CutFEM (with a standard Ersatz material approach available for comparison).
- **Self-adjoint and non-self-adjoint problems:** Both self-adjoint problems (e.g., compliance minimization) and non-self-adjoint problems (e.g., stress constraints) are supported. The adjoint operator is not derived manually but assembled automatically from the UFL symbolic representation of the constraint functional via `ufl.derivative`, leveraging FEniCSx's automatic differentiation backend.
- **Shape Optimization:** Implicit domain tracking using the Level Set method coupled with an Augmented Lagrangian Method (ALM) for constraint handling.
- **Parallel & 3D:** Fully implemented in FEniCSx to support MPI parallelization and 3D finite element formulations.
- **Extensible Architecture:** A new optimization problem is fully defined by specifying its cost and constraint functionals as UFL expressions. The adjoint operator and shape derivatives are then assembled automatically, without requiring modifications to the solvers or the optimization loop.

## Installation

OptiCut requires FEniCSx 0.11.0 and the compilation of specific C++ dependencies (`cutcells`, `runintgen`, `cutfemx`).

### Option 1: Automated Setup
A shell script is provided to automate the Conda environment creation and the C++ compilations:
```bash
git clone https://github.com/Amina-ELB/OptiCut.git
cd OptiCut
./install_opticut.sh
```

### Option 2: Manual Installation
For HPC environments without Conda, the dependencies must be built sequentially (without build isolation):
1. FEniCSx 0.11.0, MPI, PETSc, and CMake.
2. [runintgen](https://github.com/sclaus2/runintgen): `pip install --no-build-isolation --no-deps --force-reinstall ./runintgen`
3. [CutCells](https://github.com/sclaus2/cutcells): Compile the C++ core with CMake, then build the Python wrapper.
4. [CutFEMx](https://github.com/sclaus2/CutFEMx): Checkout commit `72fecb8...` and install.
5. **OptiCut**: `pip install -e .`

*(Refer to `install_opticut.sh` for exact CMake flags).*

## Quick Start (3D Coarse Demo)

To verify the parallel execution, OptiCut provides a 3D demonstration on a coarse mesh. This tutorial setup is deliberately lightweight to ensure it runs quickly on standard computers while demonstrating the MPI mechanics.

```bash
conda activate opticut-env
mpirun -n 2 python3 src/main.py src/parameters/param_compliance_3D_coarse.txt
```
The results (displacements, Von Mises stresses, and level sets) are exported in XDMF format in the `res/` directory and can be visualized using ParaView.

## Documentation

For a comprehensive overview of the mathematical formulations, numerical methods (advection, reinitialization), and a developer guide on extending the code, please refer to the official documentation:

**[Read the OptiCut Documentation](https://amina-elb.github.io/OptiCut/)**

## Contributing

We welcome contributions from the research community. Please refer to `CONTRIBUTING.md` for guidelines on submitting issues or pull requests.

## Citation

If you use this code in your academic research, please cite our upcoming JOSS publication:

```bibtex
@article{ElBachari2026OptiCut,
  title={OptiCut: Open Source FEniCSx Framework for Parallel Structural Shape Optimization},
  author={El Bachari, Amina},
  journal={Journal of Open Source Software},
  year={2026}
}
```
