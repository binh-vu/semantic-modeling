#!/bin/bash

cd /tmp
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v0.4.0
git submodule update --init
export CMAKE_PREFIX_PATH="$ANACONDA_HOME"
conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing
# setup CUDA
if [ "$ENABLE_CUDA" = "1" ]; then
    conda install -c pytorch -y magma-cuda80
    python setup.py install
else
    NO_CUDA=1 python setup.py install
fi
