from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
  name="fused_apply_rotary",
  ext_modules=[
    CUDAExtension(
      name="fused_apply_rotary",
      sources=["fuse_reorder_embedding.cpp", "fused_reorder_embedding.cu"],
      extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3","--use_fast_math"]},
    )
  ],
  cmdclass={"build_ext": BuildExtension},
)
