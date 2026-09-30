#!/bin/bash
set -e

echo "=========================================="
echo "1. Creating Conda Environment"
echo "=========================================="
conda env create -f environment.yml -y || echo "Environment already exists or failed to create."

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opticut-env

# Tie compilation variables to the environment
export CONDA_PREFIX="${CONDA_PREFIX:?activate the conda environment first}"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export PYTHONNOUSERSITE=1
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

echo "=========================================="
echo "2. Installing runintgen"
echo "=========================================="
if [ ! -d "runintgen" ]; then
    git clone https://github.com/sclaus2/runintgen.git
fi
python -m pip install --no-build-isolation --no-deps --force-reinstall ./runintgen

echo "=========================================="
echo "3. Installing CutCells"
echo "=========================================="
if [ ! -d "CutCells" ]; then
    git clone https://github.com/sclaus2/cutcells.git CutCells
fi
cd CutCells
cmake -G Ninja -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCUTCELLS_WITH_ALGOIM=OFF
cmake --build cpp/build
cmake --install cpp/build
python -m pip install --no-build-isolation --no-deps --force-reinstall ./python
cd ..

echo "=========================================="
echo "4. Installing CutFEMx"
echo "=========================================="
if [ ! -d "CutFEMx" ]; then
    git clone https://github.com/sclaus2/CutFEMx.git
    cd CutFEMx
    git checkout 422658f20355f6078f3d9a620b53cac50bbb8d27
    cd ..
fi
python -m pip install --no-build-isolation --no-deps --force-reinstall ./CutFEMx

echo "=========================================="
echo "5. Installing OptiCut"
echo "=========================================="
pip install -e .

echo "Installation complete!"
