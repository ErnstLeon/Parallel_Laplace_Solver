project('Fluid', 'cpp')

Fluid_dir = include_directories('.')
omp_dep = dependency('openmp', required: true)

mpi_dep = dependency('mpi', language: 'cpp', required: false)

executable('main_mpi', 'main.cpp',
        include_directories: Fluid_dir,
        dependencies: [mpi_dep, omp_dep],
        cpp_args: ['-std=c++20', '-O3', '-march=native', '-DMPI_FOUND']
    )

executable('main_omp', 'main.cpp',
    include_directories: Fluid_dir,
    dependencies: [omp_dep],
    cpp_args: ['-std=c++20', '-O3', '-march=native']
    )   

if mpi_dep.found()
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
