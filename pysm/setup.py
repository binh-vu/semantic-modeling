#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from Cython.Build import cythonize
from setuptools import setup, Extension

# ######################################################################################################################
# get deployment environment
# currently the only different between those deployment environments is that
# test environment need to support code coverage, therefore we have extra compiler directives
# to cython to enable coverage
env = os.environ.get("ENV", "prod")
env_os = os.environ.get("OS", "")

# ######################################################################################################################
# build ext_modules for setup function

if env == "test":
    # enable code coverage
    cythonize_kwargs = dict(compiler_directives={'linetrace': True, 'binding': True})
    extension_kwargs = dict(define_macros=[('CYTHON_TRACE', 1)])
else:
    cythonize_kwargs = dict()
    extension_kwargs = dict()

if env_os == "linux":
    extra_compile_args = ['-std=c++11']
elif env_os == "macos":
    extra_compile_args = ['-std=c++11', '-stdlib=libc++']
else:
    assert False, "Doesn't support os: %s" % env_os

exts = []
for sources in [
    ["data_structure/graph_c/graph.pyx", "data_structure/graph_c/library.cpp"],
    ["data_structure/graph_c/graph_util.pyx", "data_structure/graph_c/library.cpp"],
    ["semantic_modeling/assembling/cshare/merge_graph.pyx"],
    ["semantic_modeling/assembling/cshare/graph_explorer.pyx"],
    ["semantic_modeling/assembling/cshare/merge_planning.pyx"],
]:
    ext = Extension(
        name=sources[0].replace(".pyx", "").replace("/", "."),
        sources=["" + s for s in sources],
        language='c++',
        extra_compile_args=extra_compile_args,
        include_dirs=[
            "data_structure/graph_c"
        ],
        **extension_kwargs)
    exts.append(ext)

ext_modules = cythonize(exts, **cythonize_kwargs)

# ######################################################################################################################
# now execute setup func
setup(ext_modules=cythonize(exts))
