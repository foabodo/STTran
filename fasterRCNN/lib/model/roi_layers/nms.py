# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
import torch
from torch.utils.cpp_extension import CUDA_HOME
from fasterRCNN.lib.model import _C
# import model._C
# import os
# from site import getsitepackages
import torch
# import model
# import torchvision
# import sysconfig


# this_dir = os.path.dirname(os.path.abspath(__file__))
# model_dir = os.path.dirname(this_dir)

# model_dir = os.path.join(sysconfig.get_paths()["purelib"], "model")
# torch_dir = os.path.join(sysconfig.get_paths()["purelib"], "torch")
# lib_paths = [os.path.join(path, "model") for path in getsitepackages()]
#
# for lib_path in lib_paths:
# torch.ops.load_library(os.path.join(torch_dir, "_C.cpython-38-x86_64-linux-gnu.so"))

nms = _C.nms
# nms = model._C.nms
# nms = torch.ops._C.nms
# nms = torchvision.ops.nms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""

# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# # from ._utils import _C
# # from fasterRCNN.lib.model import _C
# import glob
# import os
#
# import torch
# from torch.utils.cpp_extension import CUDA_HOME
# from torch.utils.cpp_extension import load
#
#
# # this_dir = os.path.dirname(os.path.abspath(__file__))
# # # parent_dir = os.path.dirname(this_dir)
# # # extensions_dir = os.path.join(parent_dir, "csrc")
# # extensions_dir = os.path.dirname(this_dir)
# # print(f"extensions_dir: {extensions_dir}")
# #
# # children = os.listdir(extensions_dir)
# # print(f"children: {children}")
# #
# # # main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
# # # print(f"main_file: {main_file}")
# # #
# # # source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
# # # print(f"source_cpu: {source_cpu}")
# # #
# # # source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
# # # print(f"source_cuda: {source_cuda}")
# #
# # # sources = main_file + source_cpu
# # sources = glob.glob(os.path.join(extensions_dir, "*.so"))
# #
# # # if torch.cuda.is_available() and CUDA_HOME is not None:
# # #     sources += source_cuda
# #
# # print(f"sources: {sources}")
# this_dir = os.path.dirname(os.path.abspath(__file__))
# extensions_dir = os.path.join(os.path.dirname(this_dir), "csrc")
# print(f"extensions_dir: {extensions_dir}")
#
# children = os.listdir(extensions_dir)
# print(f"children: {children}")
#
# data_files = glob.glob(os.path.join(extensions_dir, "*.*"))
# data_files += glob.glob(os.path.join(extensions_dir, "cpu", "*.*"))
# data_files += glob.glob(os.path.join(extensions_dir, "cuda", "*.*"))
# data_files = [s.lstrip(this_dir).lstrip("/") for s in data_files]
# print(f"data_files: {data_files}")
#
# _C = load(
#     name='model',
#     sources=data_files,
#     extra_include_paths=[extensions_dir],
#     # extra_cflags=['-O2'],
#     verbose=True
# )
# nms = _C.nms
# # nms.__doc__ = """
# # This function performs Non-maximum suppresion"""
