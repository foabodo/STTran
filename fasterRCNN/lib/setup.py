# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

from collections import defaultdict
import glob
import os

from setuptools import find_packages
from distutils.core import setup
# from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


requirements = ["torch", "torchvision"]

this_dir = os.path.dirname(os.path.abspath(__file__))
print(f"this_dir: {this_dir}")
model_dir = os.path.join(this_dir, "model")
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

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(csrc_dir, s) for s in sources]
    print(f"sources: {sources}")
    sources = [s.replace(this_dir + "/", "") for s in sources]
    print(f"lstripped sources: {sources}")


    include_dirs = [csrc_dir]

    ext_modules = [
        extension(
            "model._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            # library_dirs=[model_dir],
            # libraries=[os.path.join(model_dir, "_C.cpython-38-x86_64-linux-gnu.so")]
        )
    ]

    return ext_modules


data_files = glob.glob(os.path.join(csrc_dir, "*.*"))
data_files += glob.glob(os.path.join(csrc_dir, "cpu", "*.*"))
data_files += glob.glob(os.path.join(csrc_dir, "cuda", "*.*"))
print(f"DATA_FILES = {data_files}")

header_files = [f for f in data_files if f.rsplit(".")[-1] == "h"]
print(f"HEADER_FILES = {header_files}")

# tuples = defaultdict(list)
# for f in data_files:
#     tuples[os.path.dirname(f)].append(f)
# tuples = [(k, v) for k, v in tuples.items()]

# data_files = [s.lstrip(this_dir).lstrip("/") for s in data_files]

print(f"data_files: {data_files}")
print([s.lstrip(model_dir).lstrip("/") for s in data_files])
setup(
    name="faster_rcnn",
    version="0.1",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),# + ["model"],
    # install_requires=requirements,
    # libraries=get_extensions(),
    ext_modules=get_extensions(),
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
