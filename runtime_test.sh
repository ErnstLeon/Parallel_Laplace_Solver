#!/bin/bash

rm -rf runtime_omp.dat

for i in 10 50 100 500 1000 5000 10000 20000;
do
   # IFS=$'\n' OUTPUT=( $(mpirun -np 6 --hostfile hostfile ./main_mpi -x_dim $i -y_dim $i -nthreads 6 ) )
    IFS=$'\n' OUTPUT=( $(./main_omp -x_dim $i -y_dim $i -nthreads 6 ) )

    (echo "$i ${OUTPUT[0]}") >> runtime_omp.dat
    echo $i
done