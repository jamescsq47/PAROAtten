import os
import subprocess
from packaging.version import parse, Version
from typing import List, Set
import warnings

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

HAS_SM80 = False
HAS_SM86 = False
HAS_SM89 = False
HAS_SM90 = False

# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"8.0"}

# Compiler flags.
CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    "--threads=8",
    "-Xptxas=-v",
    "-diag-suppress=174", # suppress the specific warning
]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

def get_torch_arch_list() -> Set[str]:
    # TORCH_CUDA_ARCH_LIST can have one or more architectures,
    # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
    # compiler to additionally include PTX code that can be runtime-compiled
    # and executed on the 8.6 or newer architectures. While the PTX code will
    # not give the best performance on the newer architectures, it provides
    # forward compatibility.
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    # List are separated by ; or space.
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}.")
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}.")
    return arch_list

# First, check the TORCH_CUDA_ARCH_LIST environment variable.
compute_capabilities = get_torch_arch_list()
if not compute_capabilities:
    # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
    # GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            raise RuntimeError(
                "GPUs with compute capability below 8.0 are not supported.")
        compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    raise RuntimeError("No GPUs found. Please specify the target GPU architectures or build on a machine with GPUs.")

# Validate the NVCC CUDA version.
# if nvcc_cuda_version < Version("12.0"):
#     raise RuntimeError("CUDA 12.0 or higher is required to build the package.")
if nvcc_cuda_version < Version("12.4") and any(cc.startswith("8.9") for cc in compute_capabilities):
    raise RuntimeError(
        "CUDA 12.4 or higher is required for compute capability 8.9.")
if nvcc_cuda_version < Version("12.3") and any(cc.startswith("9.0") for cc in compute_capabilities):
    if any(cc.startswith("9.0") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 12.3 or higher is required for compute capability 9.0.")

# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    num = capability[0] + capability[2]
    if num == "80":
        HAS_SM80 = True
    elif num == "86":
        HAS_SM86 = True
    elif num == "89":
        HAS_SM89 = True
    elif num == "90":
        HAS_SM90 = True
        num = num + "a" # convert sm90 to sm9a
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

ext_modules = []

if HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90:
    qattn_extension = CUDAExtension(
        name="paroattention._qattn_sm80",
        sources=[
            "csrc/qattn/pybind_sm80.cpp",
            "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
            # "csrc/qattn/qk_int_sv_int4_cuda_sm80.cu",
            "csrc/qattn/qk_int_sv_int8_cuda_sm80.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(qattn_extension)

# Fused kernels.
rope_extension = CUDAExtension(
    name="paroattention._rope",
    sources=["csrc/rope/pybind.cpp", "csrc/rope/rope.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(rope_extension)

# Fused kernels.
fused_extension = CUDAExtension(
    name="paroattention._fused",
    sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_extension)

setup(
    name='paroattention', 
    version='0.0.1',  
    author='PAROAttention team',
    description='PAROAttention code adapted from SageAttention.',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown', 
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)