#project('Fluid', ['cpp', 'cuda'])  # Enable both C++ and CUDA languages; requires nvcc at configure time
project('Fluid', 'cpp')          # Enable only C++ language; no CUDA support, no nvcc required

Fluid_dir = include_directories('../include/')

omp_dep = dependency('openmp', required: true)
mpi_dep = dependency('mpi', language: 'cpp', required: false)
cuda_dep = dependency('cuda', required: false)

cpp_flags = ['-std=c++20', '-O3', '-march=native']

# MPI + OpenMP build
executable('test_mpi', 'test_mpi.cpp',
        include_directories: Fluid_dir,
        dependencies: [mpi_dep, omp_dep],
        cpp_args: cpp_flags + ['-DMPI_FOUND']
    )

# OpenMP-only build
executable('test_omp', 'test_omp.cpp',
    include_directories: Fluid_dir,
    dependencies: [omp_dep],
    cpp_args: cpp_flags
    )   

# CUDA build
#executable('test_cuda', 'test_cuda.cu',
#    include_directories: Fluid_dir,
#    dependencies: [cuda_dep],
#    cpp_args: cpp_flags,
#    cuda_args: ['-arch=sm_75', '-Xcompiler=-O3,-march=native']
#)
