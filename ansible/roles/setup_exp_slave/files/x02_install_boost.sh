#!/bin/bash

cd /tmp
wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz
tar -xf boost_1_67_0.tar.gz
sudo mv boost_1_67_0/boost /usr/local/include/boost