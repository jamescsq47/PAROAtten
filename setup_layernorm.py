# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_reorder_ln",  # 对应 TORCH_EXTENSION_NAME
    ext_modules=[
        CUDAExtension(
            name="fused_reorder_ln",            # Python import 模块名
            sources=[
                "fuse_reorder_layernorm.cpp",  # C++ 封装
                "fused_reorder_layernorm.cu",   # CUDA 实现
            ],
            extra_compile_args={
                "cxx": ["-O3"],                # C++ 优化
                "nvcc": ["-O3", "--use_fast_math"],  # NVCC 优化选项
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension  # 将 `python setup.py build_ext` 定向到 PyTorch 的构建系统
    },
)
