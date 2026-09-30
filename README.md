# OptiCut: Parallel Structural Shape Optimization in FEniCSx

[![CI Build](https://github.com/Amina-ELB/OptiCut/actions/workflows/tests.yml/badge.svg)](https://github.com/Amina-ELB/OptiCut/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://amina-elb.github.io/OptiCut/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **OptiCut** is a cutting-edge, parallel-enabled FEniCSx framework for designing lightweight and high-performance components. It implements advanced optimization algorithms based on immersed boundary methods for numerical simulations in engineering.

<p align="center">
  <img src="./doc/images/opticut-logo-v2.png" alt="OptiCut Logo">
</p>

---

## Framework Overview

OptiCut is designed for engineers and researchers in advanced numerical simulation. Built upon the **FEniCSx** project, it leverages modern capabilities for efficient, **distributed parallel computing (MPI)**.

It provides a powerful solution for structural design by integrating advanced numerical methods:
1. **Level Set Method**: For seamless geometry representation and topological evolution.
2. **Cut Finite Element Method (CutFEM)**: For solving equations on non-conforming background meshes without the need for constant remeshing.
3. **Ersatz Material Approach**: For simplified modeling and fast preliminary design phases.

---

## Key Features

OptiCut is built with **modularity** and **adaptability** in mind:

- **Full 3D Capability**: The core implementation handles both 2D and 3D geometries, enabling the optimization of complex, real-world structural components.
- **Parallel Computing (MPI)**: Inherits FEniCSx's robust performance on large-scale problems and distributed memory systems.
- **High Extensibility**: Features a highly modular `Problem` class architecture. By simply overriding this class, users can easily adapt the framework to solve entirely new optimization problems beyond standard compliance.
- **Rich Post-Processing**: Automatically exports optimization states, constraints, and physical fields (displacement, Von Mises stress) in XDMF format for seamless visualization in ParaView.

---

## Installation

Installing OptiCut requires a properly configured FEniCSx 0.11.0 environment and the compilation of several C++ dependencies (CutCells, runintgen, CutFEMx). 

### Option 1: Automated Installation (Recommended)
We provide an automated script that creates the Conda environment and handles all C++ compilations for you.

```bash
# Clone the repository
git clone https://github.com/Amina-ELB/OptiCut.git
cd OptiCut

# Run the automated installation script
./install_opticut.sh
```

### Option 2: Manual Installation (For HPC or Advanced Users)
If you are working on a cluster without Conda or prefer to build the environment manually, you must follow the correct build order (without build isolation):

1. **Environment:** Ensure FEniCSx 0.11.0, MPI, PETSc, and `cmake` are installed and loaded.
2. **Install runintgen:** Clone [runintgen](https://github.com/sclaus2/runintgen) and install via `pip install --no-build-isolation --no-deps --force-reinstall ./runintgen`.
3. **Install CutCells:** Clone [CutCells](https://github.com/sclaus2/cutcells). Compile the C++ core using CMake (`cmake -S cpp -B cpp/build ...`), install it, and build the Python wrapper (`pip install --no-build-isolation --no-deps --force-reinstall ./python`).
4. **Install CutFEMx:** Clone [CutFEMx](https://github.com/sclaus2/CutFEMx) (checkout commit `72fecb8...`) and install via `pip install --no-build-isolation --no-deps --force-reinstall ./CutFEMx`.
5. **Install OptiCut:** Finally, install this framework using `pip install -e .`.

*(Please refer to `install_opticut.sh` for the exact CMake flags and environment variables required).*

---

## Quick Start (Minimal Example)

Once installed, you can quickly run a 3D structural shape optimization example in parallel (e.g., using 2 MPI processes). Open a new terminal, activate the environment, and execute from the root of the repository:

```bash
conda activate opticut-env
mpirun -n 2 python3 src/main.py src/parameters/param_compliance_3D.txt
```

This will run the compliance minimization demo using the provided parameter file. Output files will be saved in the `res/` directory.

---

## Documentation

The **full documentation**, including parallel execution guidance, tutorials, and underlying mathematical theory, is available on the official website:

**[Access the Complete OptiCut Technical Documentation](https://amina-elb.github.io/OptiCut/)**

---

## Demonstration Examples

The framework comes with two main demonstrations illustrating its powerful applications:

1. **Structural Compliance Minimization (Stiffness Optimization)**: Utilizes the Ersatz and CutFEMx methods within a parallel setup to optimize the overall stiffness of structures.
2. **Minimization of the $L^p$ Norm of the von Mises Stress (Strength)**: Applies the CutFEMx method to design components with improved stress resistance by minimizing stress peaks.

---

## Contributing

This project is **Open Source**. We welcome contributions, bug reports, and feature requests. Please refer to the `CONTRIBUTING.md` file (or the "Contribution" section in the documentation) for more details.

---

## Citation

If you use OptiCut in your research, please cite our upcoming JOSS paper:

```bibtex
@article{ElBachari2026OptiCut,
  title={OptiCut: Open Source FEniCSx Framework for Parallel Structural Shape Optimization},
  author={El Bachari, Amina},
  journal={Journal of Open Source Software},
  year={2026}
}
```
*(Note: Citation details will be updated upon JOSS publication).*
