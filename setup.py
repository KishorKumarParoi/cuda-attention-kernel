from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

ext_modules = []

if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension(
            "flash_attn_cuda",
            [
                "csrc/binding.cpp",
                "csrc/kernels/flash_attn_v2.cu",
                "csrc/kernels/flash_attn_bwd.cu",
                "csrc/kernels/flash_attn_cpu.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ],
            },
        )
    )
else:
    ext_modules.append(
        CppExtension(
            "flash_attn_cpu",
            [
                "csrc/binding.cpp",
                "csrc/kernels/flash_attn_cpu.cpp",
            ],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    )

setup(
    name="flash_attn_kernel",
    version="1.0.0",
    author="Kishor Kumar Paroi <1703053@student.ruet.ac.bd>",
    description="FlashAttention-2 Custom CUDA Kernel with Tiled Online Softmax",
    packages=["flash_attn"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
