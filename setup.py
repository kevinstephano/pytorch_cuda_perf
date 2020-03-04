from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import sys

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

element_wise_ext = CUDAExtension(
                        name='element_wise_ext',
                        sources=['csrc/element_wise.cpp', 'csrc/element_wise_cuda.cu'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                                'nvcc':['--gpu-architecture=compute_70','--gpu-code=sm_70','-O3','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', "--expt-relaxed-constexpr"]
                        }
)
"""
reduction_ext = CUDAExtension(
                        name='reduction_ext',
                        sources=['csrc/reduction.cpp', 'csrc/reduction_cuda.cu'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                                'nvcc':['--gpu-architecture=compute_70','--gpu-code=sm_70','-O3','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', "--expt-relaxed-constexpr"]
                        }
)
"""

setup(
    name='Cuda Perf',
    version='0.1.0',
    description='Test various Cuda code in pytorch.',
    packages=find_packages(),
    ext_modules=[element_wise_ext],
    cmdclass={
                'build_ext': BuildExtension
    },
    test_suite='tests',
)

