# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

from collections import defaultdict
import glob
import numpy
import os

import torch
from setuptools import find_packages
from distutils.core import setup
from Cython.Build import cythonize
# from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


requirements = ["torch", "torchvision"]

sttran_dir = os.path.dirname(os.path.abspath(__file__))
print(f"sttran_dir: {sttran_dir}")
faster_rcnn_dir = os.path.join(sttran_dir, "fasterRCNN")
print(f"faster_rcnn_dir: {faster_rcnn_dir}")
lib_dir = os.path.join(faster_rcnn_dir, "lib")
print(f"lib_dir: {lib_dir}")
model_dir = os.path.join(lib_dir, "model")
print(f"model_dir: {model_dir}")
csrc_dir = os.path.join(model_dir, "csrc")
print(f"csrc_dir: {csrc_dir}")


def get_extensions():
    main_file = glob.glob(os.path.join(csrc_dir, "*.cpp"))
    print(f"main_file: {main_file}")
    source_cpu = glob.glob(os.path.join(csrc_dir, "cpu", "*.cpp"))
    print(f"source_cpu: {source_cpu}")
    source_cuda = glob.glob(os.path.join(csrc_dir, "cuda", "*.cu"))
    print(f"source_cuda: {source_cuda}")

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    print(f"WARNING: Is CUDA available? : {torch.cuda.is_available()}")
    print(f"WARNING: CUDA_HOME: {CUDA_HOME}")

    sources = [os.path.join(csrc_dir, s) for s in sources]
    print(f"sources: {sources}")
    sources = [s.replace(sttran_dir + "/", "") for s in sources]
    print(f"lstripped sources: {sources}")


    include_dirs = [csrc_dir]

    ext_modules = [
        extension(
            "fasterRCNN.lib.model._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            # library_dirs=[model_dir],
            # libraries=[os.path.join(model_dir, "_C.cpython-38-x86_64-linux-gnu.so")]
        )
    ] + cythonize(['lib/draw_rectangles/draw_rectangles.pyx', 'lib/fpn/box_intersections_cpu/bbox.pyx'])

    return ext_modules


data_files = glob.glob(os.path.join(csrc_dir, "*.*"))
data_files += glob.glob(os.path.join(csrc_dir, "cpu", "*.*"))
data_files += glob.glob(os.path.join(csrc_dir, "cuda", "*.*"))
print(f"DATA_FILES = {data_files}")

# header_files = [f for f in data_files if f.rsplit(".")[-1] == "h"]
# print(f"HEADER_FILES = {header_files}")

# tuples = defaultdict(list)
# for f in data_files:
#     tuples[os.path.dirname(f)].append(f)
# tuples = [(k, v) for k, v in tuples.items()]

# data_files = [s.lstrip(this_dir).lstrip("/") for s in data_files]

print(f"data_files: {data_files}")
print([s.replace(model_dir + "/", "") for s in data_files])
setup(
    name="sttran",
    version="0.1",
    description="scene graph detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),# + ["model"],
    # install_requires=requirements,
    # libraries=get_extensions(),
    ext_modules=get_extensions(),
    include_dirs=[numpy.get_include()],
    # ext_package="model",
    cmdclass={"build_ext": BuildExtension},
    # package_dir={"_C": "model"},
    # package_data={"_C": ["csrc/*", "src/cpu/*", "csrc/cuda/*"]},
    python_requires=">=3.8",
    # data_files=tuples,
    # libraries=["_C"],
    # package_dir={"model": "model"},
    # package_data={"model": [s.lstrip(model_dir).lstrip("/") for s in data_files]},
    # headers=header_files,
    # include_package_data=True
)
