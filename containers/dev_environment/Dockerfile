FROM ubuntu:16.04

RUN apt-get update

# INSTALL kcov for code coverage!
RUN apt-get install -y --fix-missing pkg-config
RUN apt-get install -y zlib1g wget libcurl4-openssl-dev libelf-dev libdw-dev cmake cmake-data g++ binutils-dev \
                       libiberty-dev zlib1g-dev python-minimal git

RUN cd /tmp && \
    wget https://github.com/SimonKagstrom/kcov/archive/master.tar.gz && \
    tar xzf master.tar.gz && \
    cd kcov-master && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# INSTALL RUST
RUN apt-get install -y curl
RUN curl https://sh.rustup.rs -sSf > install_rust_script.sh && sh install_rust_script.sh -y && rm install_rust_script.sh

# INSTALL DEPENDENCIES: ANACONDA, PyTorch, SparseHash, Boost
RUN cd /tmp && \
    wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh && \
    bash Anaconda3-5.2.0-Linux-x86_64.sh -u -b

ENV PATH="/root/anaconda3/bin:/root/.cargo/bin/:${PATH}"

ADD x00_install_pytorch.sh /tmp
RUN cd /tmp && bash x00_install_pytorch.sh

ADD x01_install_sparsehash.sh /tmp
RUN cd /tmp && bash x01_install_sparsehash.sh

ADD x02_install_boost.sh /tmp
RUN cd /tmp && bash x02_install_boost.sh

RUN apt install -y libzmq5-dev

# SETUP workspace
ADD source_code /tmp/source_code
RUN cd /tmp/source_code && \
    cargo build

RUN mkdir -p /workspace/semantic-modeling
WORKDIR /workspace/semantic-modeling