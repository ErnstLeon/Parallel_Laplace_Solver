#!/bin/bash

rm -rf runtime.dat

for i in $(seq 10 -1 1)
do
    for c in $(seq 1 1 1)
    do
        IFS=$'\n' OUTPUT=( $(./builddir/main_omp -x_dim 1500 -y_dim 1500 -nthreads $i 2>/dev/null) )

        (echo "$i ${OUTPUT[0]}") >> runtime.dat
    done
done