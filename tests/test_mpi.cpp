#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>

#include <matrix_type.hpp>

#include <mpi.h>

#ifdef USE_DOUBLE
#define MPI_TYPE  MPI_DOUBLE
#else
#define MPI_TYPE  MPI_FLOAT
#endif

#include <flow_type_mpi.hpp>

int main(int argc, char **argv){

#ifdef USE_DOUBLE
  using T = double;
#else
  using T = float;
#endif

  int nthreads = 1;
  int x_dim = 100;
  int y_dim = 100;

  for(char ** arg = argv; *arg != NULL; ++arg){
    if(strcmp(*arg, "-nthreads") == 0){
      nthreads = atoi(*(arg + 1)); 
      ++arg;
    }
    if(strcmp(*arg, "-x_dim") == 0){
      x_dim = atoi(*(arg + 1)); 
      ++arg;
    }
    if(strcmp(*arg, "-y_dim") == 0){
      y_dim = atoi(*(arg + 1)); 
      ++arg;
    }
  }

  std::string output_filename = "velocities_out_mpi.dat";

  matrix<T> init{x_dim, y_dim, static_cast<T>(0)};

  for(int i = 0; i <= x_dim / 4; ++i) init(i, 0) = T(14);
  for(int i = x_dim / 4 + 1; i < x_dim / 2; ++i) init(i, 0) = T(14) - (T(i) - T(x_dim) / T(4)) * T(56) / T(x_dim);

  for(int i = y_dim / 2 + 1; i < 3 * y_dim / 4; ++i) init(x_dim - 1, i) = (T(i) - T(y_dim) / T(2)) * T(28) / T(y_dim);
  for(int i = 3 * y_dim / 4; i < y_dim; ++i) init(x_dim - 1, i) = T(7);

  for(int i = 0; i < x_dim; ++i) init(i, y_dim - 1) = T(7);

  for(int i = 0; i <= y_dim / 2; ++i) init(0, i) = T(14);
  for(int i = y_dim / 2 + 1; i < 3 * y_dim / 4; ++i) init(0, i) = T(7) + (T(3) * T(y_dim) / T(4) - T(i)) * T(28) / T(y_dim);
  for(int i = 3 * y_dim / 4; i < y_dim; ++i) init(0, i) = T(7);


  MPI_Init(&argc, &argv);

  flow<T> psi {std::move(init), 10000};
  psi.solve(nthreads);

  std::cout << psi.get_runtime() << std::endl;
  psi.save(output_filename);

  psi.cleanup();
  MPI_Finalize();

}
