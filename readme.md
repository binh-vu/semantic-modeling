# Installation

### Install aten library from pytorch

```
git clone https://github.com/pytorch/pytorch.git
git checkout v0.4.0
```

```
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing

# Add LAPACK support for the GPU
conda install -c pytorch magma-cuda80 # or magma-cuda90 if CUDA 9
```

### Compile cython

```
ENV=prod OS=macos python setup.py build_ext --inplace
```

### Run ansible to setup server on remote servers

```
cd containers/ansible
ansible-playbook -i inventory.ini setup-vm.yml
```

# Instruction

To run experiments, run scripts in bin/experiments folder.

The experiment results are in folder experiments