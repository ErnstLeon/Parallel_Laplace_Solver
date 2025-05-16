project('Fluid', ['cpp','cuda'])
#project('Fluid', 'cpp')

Fluid_dir = include_directories('./include')

omp_dep = dependency('openmp', required: true)
mpi_dep = dependency('mpi', language: 'cpp', required: false)
cuda_dep = dependency('cuda', required: false)

# CUDA build
if cuda_dep.found()
    executable('main_cuda', 'main.cu',
        include_directories: Fluid_dir,
        dependencies: [cuda_dep],
        cpp_args: ['-std=c++20', '-O3', '-march=native'],
        cuda_args: ['-arch=sm_75', '-Xcompiler=-O3,-march=native']
    )
endif

# MPI + OpenMP build
if mpi_dep.found()
    executable('main_mpi', 'main.cpp',
        include_directories: Fluid_dir,
        dependencies: [mpi_dep, omp_dep],
        cpp_args: ['-std=c++20', '-O3', '-march=native', '-DMPI_FOUND']
    )
endif

# OpenMP-only build
executable('main_omp', 'main.cpp',
    include_directories: Fluid_dir,
    dependencies: [omp_dep],
    cpp_args: ['-std=c++20', '-O3', '-march=native']
    )   

if cuda_dep.found()
    executable('main', 'main.cu',
    include_directories: Fluid_dir,
    dependencies: [cuda_dep],
    cpp_args: ['-std=c++20', '-O3', '-march=native'],
    cuda_args: ['-arch=sm_75', '-Xcompiler=-O3,-march=native']
    )
elif mpi_dep.found()
    executable('main', 'main.cpp',
        include_directories: Fluid_dir,
        dependencies: [mpi_dep, omp_dep],
        cpp_args: ['-std=c++20', '-O3', '-march=native', '-DMPI_FOUND']
    )
else
    executable('main', 'main.cpp',
    include_directories: Fluid_dir,
    dependencies: [omp_dep],
    cpp_args: ['-std=c++20', '-O3', '-march=native',]
    )   
endif
