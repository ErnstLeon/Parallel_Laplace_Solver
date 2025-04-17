project('Fluid', 'cpp', 'cuda')

Fluid_dir = include_directories('./include')

omp_dep = dependency('openmp', required: true)
mpi_dep = dependency('mpi', language: 'cpp', required: false)
cuda_dep = dependency('cuda', required: true)

# MPI + OpenMP build
executable('main_mpi', 'main.cpp',
        include_directories: Fluid_dir,
        dependencies: [mpi_dep, omp_dep],
        cpp_args: ['-std=c++20', '-O3', '-march=native', '-DMPI_FOUND']
    )

# OpenMP-only build
executable('main_omp', 'main.cpp',
    include_directories: Fluid_dir,
    dependencies: [omp_dep],
    cpp_args: ['-std=c++20', '-O3', '-march=native']
    )   

# CUDA build
executable('main_cuda', 'main.cu',
    include_directories: Fluid_dir,
    dependencies: [cuda_dep],
    cpp_args: ['-std=c++20', '-O3', '-march=native'],
    cuda_args: ['-arch=sm_75']
)

if cuda_dep.found()
    executable('main', 'main.cu',
    include_directories: Fluid_dir,
    dependencies: [cuda_dep],
    cpp_args: ['-std=c++20', '-O3', '-march=native'],
    cuda_args: ['-arch=sm_75']
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
