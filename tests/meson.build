project('Fluid', 'cpp')# 'cuda')

Fluid_dir = include_directories('../include')

omp_dep = dependency('openmp', required: true)
mpi_dep = dependency('mpi', language: 'cpp', required: false)
#cuda_dep = dependency('cuda', required: true)

# MPI + OpenMP build
executable('test_mpi', 'test_mpi.cpp',
        include_directories: Fluid_dir,
        dependencies: [mpi_dep, omp_dep],
        cpp_args: ['-std=c++20', '-O3', '-march=native', '-DMPI_FOUND']
    )

# OpenMP-only build
executable('test_omp', 'test_omp.cpp',
    include_directories: Fluid_dir,
    dependencies: [omp_dep],
    cpp_args: ['-std=c++20', '-O3', '-march=native']
    )   

# CUDA build
#executable('test_cuda', 'test_cuda.cu',
#    include_directories: Fluid_dir,
#    dependencies: [cuda_dep],
#    cpp_args: ['-std=c++20', '-O3', '-march=native'],
#    cuda_args: ['-arch=sm_75', '-Xcompiler=-O3,-march=native']
#)
