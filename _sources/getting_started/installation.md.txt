# Installation

OptiCut relies heavily on **FEniCSx** (v0.11.0) and **CutFEMx** for solving partial differential equations. Due to complex C++ and MPI dependencies, we provide a streamlined Conda-based installation script.

## Prerequisites

- A Linux or macOS operating system (Windows users should use WSL2).
- [Miniforge](https://github.com/conda-forge/miniforge) or Miniconda installed on your system.

## Option 1: Automated Installation (Recommended)

To ensure strict reproducibility and perfectly matching C++ compilers, we provide an automated bash script. This script will create a Conda environment named `opticut-env`, fetch FEniCSx 0.11.0, compile the required CutFEMx dependencies (`runintgen` and `CutCells`), and finally install OptiCut.

```bash
# Clone the repository
git clone https://github.com/Amina-ELB/OptiCut.git
cd OptiCut

# Run the automated installation script
./install_opticut.sh
```

## Option 2: Docker (Recommended for Reproducibility)

A pre-configured Docker image is provided for maximum portability. This option requires no prior installation of FEniCSx, Conda, or any C++ toolchain — the container provides a fully self-contained environment.

```bash
# Clone the repository
git clone https://github.com/Amina-ELB/OptiCut.git
cd OptiCut

# Build the Docker image
docker build -t opticut .

# Run an optimization (mounts the current directory into the container)
docker run --rm -v $(pwd):/workspace opticut \
    python3 src/main.py src/parameters/param_compliance.txt
```

This approach is particularly useful for evaluation, continuous integration, or sharing results across different computing environments.

## Option 3: Manual Installation (For HPC Clusters)

If you are working on a High Performance Computing (HPC) cluster where Conda is unavailable, or if you prefer to build the environment manually, you must strictly respect the build order without build isolation.

First, define your installation prefix (the directory where FEniCSx and its dependencies are installed):

```bash
export PREFIX=/path/to/your/install  # e.g., $HOME/.local or a module-loaded path
```

Then install each dependency in order:

1. **Base Environment:** Ensure FEniCSx 0.11.0, MPI, PETSc, and `cmake` are installed and loaded into your environment.
2. **runintgen:**
   ```bash
   git clone https://github.com/sclaus2/runintgen.git
   python -m pip install --no-build-isolation --no-deps --force-reinstall ./runintgen
   ```
3. **CutCells:**
   ```bash
   git clone https://github.com/sclaus2/cutcells.git CutCells
   cd CutCells
   cmake -G Ninja -S cpp -B cpp/build \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX="$PREFIX" \
       -DCMAKE_PREFIX_PATH="$PREFIX" \
       -DCUTCELLS_WITH_ALGOIM=OFF
   cmake --build cpp/build
   cmake --install cpp/build
   python -m pip install --no-build-isolation --no-deps --force-reinstall ./python
   cd ..
   ```
4. **CutFEMx:**
   ```bash
   git clone https://github.com/sclaus2/CutFEMx.git
   cd CutFEMx
   git checkout 422658f20355f6078f3d9a620b53cac50bbb8d27
   cd ..
   python -m pip install --no-build-isolation --no-deps --force-reinstall ./CutFEMx
   ```
5. **OptiCut:**
   ```bash
   pip install -e .
   ```

## Testing the Installation

To verify that the environment is perfectly configured, you can run the built-in test suite:

```bash
conda activate opticut-env
pytest tests/
```

If all tests pass, your environment is ready for shape optimization.

:::{admonition} Optional: Documentation Dependencies
:class: note

If you wish to build the HTML documentation locally (as seen here), you must install Sphinx and its associated extensions:

```bash
pip install sphinx furo nbsphinx sphinxcontrib-bibtex myst-parser sphinxcontrib-mermaid
```
:::
