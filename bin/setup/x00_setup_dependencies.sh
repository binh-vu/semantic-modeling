#!/bin/bash

set -e
set -o pipefail

# setup global variables
CURRENT_DIR=$(pwd)

# check to make sure anaconda has been installed before.
if [ -z "$ANACONDA_HOME" ]; then
    echo "Anaconda need to be installed before"
    exit 1
fi

# make sure user input correct environment
if [ "$ENV" != "linux" ]; then
    if [ "$ENV" != "macos" ]; then
        echo "Invalid environment, has to be either linux or macos. Get $ENV instead"
        exit 1
    fi
fi

# install libGL first (for matplotlib python)
python -mplatform | grep -qi ubuntu && apt-get install -y libgl1-mesa-dev
python -mplatform | grep -qi debian && apt-get install -y libgl1-mesa-dev
python -mplatform | grep -qi centos && apt-get install -y mesa-libGL

# installing pytorch
cd /tmp
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v0.4.0
git submodule update --init
export CMAKE_PREFIX_PATH="$ANACONDA_HOME"
if [ "$ENV" == "linux" ]; then
    conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing
    # setup CUDA
    if [ "$ENABLE_CUDA" = "1" ]; then
        conda install -c pytorch -y magma-cuda80
        python setup.py install
    else
        NO_CUDA=1 python setup.py install
    fi

    # setup sparsehash
    cd /tmp
    git clone https://github.com/sparsehash/sparsehash.git
    cd sparsehash && ./configure && sudo make install

    # setup boost
    cd /tmp
    wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz
    tar -xf boost_1_67_0.tar.gz
    sudo mv boost_1_67_0/boost /usr/local/include/boost
else
    conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
    MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
fi

# install ctensor
cd "$CURRENT_DIR/gmtk/ctensor"
if [ -d "build" ]; then
    rm -r build
fi
mkdir build
cd build
cmake ..
make
sudo cp libctensor.* /usr/local/lib/

cd "$CURRENT_DIR/pysm"
OS="$ENV" ENV=prod python setup.py build_ext --inplace