#project('Fluid', ['cpp', 'cuda'])  # Enable both C++ and CUDA languages; requires nvcc at configure time
project('Fluid', 'cpp')          # Enable only C++ language; no CUDA support, no nvcc required

Fluid_dir = include_directories('../include')

omp_dep = dependency('openmp', required: true)
mpi_dep = dependency('mpi', language: 'cpp', required: false)
cuda_dep = dependency('cuda', required: false)

cpp_flags = ['-std=c++20', '-O3', '-march=native']

if cuda_dep.found()
    message('Configuring build: CUDA version enabled')
    message('./main -x_dim <value> -y_dim <value>')
elif mpi_dep.found()
    message('Configuring build: MPI + OpenMP version enabled')
    message('mpirun -np <num_processes> --hostfile <hostfile_path> ./main -x_dim <value> -y_dim <value> -nthreads <num_threads>')
else
    message('Configuring build: OpenMP-only version enabled')
    message('./main -x_dim <value> -y_dim <value> -nthreads <num_threads>')
endif

# CUDA build
if cuda_dep.found()
    executable('main_cuda', 'main.cu',
        include_directories: Fluid_dir,
        dependencies: [cuda_dep],
        cpp_args: cpp_flags,
        cuda_args: ['-arch=sm_75', '-Xcompiler=-O3,-march=native']
    )
endif

# MPI + OpenMP build
if mpi_dep.found()
    executable('main_mpi', 'main.cpp',
        include_directories: Fluid_dir,
        dependencies: [mpi_dep, omp_dep],
        cpp_args: cpp_flags + ['-DMPI_FOUND']
    )
endif

# OpenMP-only build
executable('main_omp', 'main.cpp',
    include_directories: Fluid_dir,
    dependencies: [omp_dep],
    cpp_args: cpp_flags
    )   

if cuda_dep.found()
    executable('main', 'main.cu',
    include_directories: Fluid_dir,
    dependencies: [cuda_dep],
    cpp_args: cpp_flags,
    cuda_args: ['-arch=sm_75', '-Xcompiler=-O3,-march=native']
    )
elif mpi_dep.found()
    executable('main', 'main.cpp',
        include_directories: Fluid_dir,
        dependencies: [mpi_dep, omp_dep],
        cpp_args: cpp_flags + ['-DMPI_FOUND']
    )
else
    executable('main', 'main.cpp',
    include_directories: Fluid_dir,
    dependencies: [omp_dep],
    cpp_args: cpp_flags
    )   
endif
