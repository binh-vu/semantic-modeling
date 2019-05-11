#!/bin/bash

cd /tmp
git clone https://github.com/sparsehash/sparsehash.git
cd sparsehash && ./configure && make install
