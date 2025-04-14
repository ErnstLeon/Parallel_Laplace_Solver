#ifndef FLOW_TYPE_H
#define FLOW_TYPE_H

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>

#include <matrix_type.hpp>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    throw std::runtime_error("Cuda runtime error: " + std::string(cudaGetErrorString(result)));
  }
  return result;
}

template<typename T>
__global__ void cuda_jacobi(T* device_matrix_1, T* device_matrix_2, T* diff, T epsilon, int max_iter, int x_dim, int y_dim){

    int x_stride = blockDim.x * gridDim.x;
    int y_stride = blockDim.y * gridDim.y;

    int iter{};

    do{
        T local_diff = static_cast<T>(0);

        int old = iter % 2;
        int next = (iter + 1) % 2;

        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < x_dim; x += x_stride) 
        {
            for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < y_dim; y += y_stride) 
            {
                int idx = y_dim * x + y;
                device_matrix_1[idx] = idx;

                
            }
        }

        iter++;
        atomicAdd(diff, local_diff);

    }while (*diff >= epsilon * epsilon && iter < max_iter);

}

/*
    Class containing routines for solving Laplace equation on a 2D grid with boundary conditions

    Template typename T : type used for calculation (double or float)

    Solves the Laplace equation ((d/dx)^2 + (d/dy)^2)phi(x,y) = 0
    using jacobi iterations with cuda. 

    Returns the x- and y-velocity for each point in a file (excluding the boundary conditions).
    File is binary with ordered like v_x(0, 0), v_y(0, 0), v_x(0, 1), v_y(0, 1) ...
*/
template<typename T>
class flow {

    using Matrix = matrix<T>;

private:

    // Two matrices, storing the local values of phi on the grid
    // containing old and new values during the iterative solution
    std::array<Matrix, 2> values{};

    // Two matrices to store the local velocities for each point on the grid;
    // first one to hold x-velocity, second to hold y-velocity
    std::array<Matrix, 2> velocity{};

    // Precision and max iteration count for iterative solution
    T epsilon{static_cast<T>(1e-8)};
    int max_iter = static_cast<int>(10000);

    // stores id of the value matrix that contains the final result 
    int result_id{};

    std::chrono::duration<double> runtime{};

public:

    // Constructor with matrix containing the boundary conditions in the outer
    // column and row and initial values in the inner matrix
    flow(const Matrix& init) : 
        values{Matrix(init), Matrix(init)}, 
        velocity{Matrix(init.size_x(), init.size_y()), Matrix(init.size_x(), init.size_y())} {}

    // Constructor with rvalue matrix containing the boundary conditions in the outer
    // column and row and initial values in the inner matrix 
    flow(Matrix&& init) noexcept : 
        values{Matrix(init), Matrix(std::move(init))},
        velocity{Matrix(init.size_x(), init.size_y()), Matrix(init.size_x(), init.size_y())} {}

    // Constructor with matrix containing the boundary conditions in the outer
    // column and row and initial values in the inner matrix and given epsilon
    flow(const Matrix& init, T epsilon) : 
        values{Matrix(init), Matrix(init)},
        velocity{Matrix(init.size_x(), init.size_y()), Matrix(init.size_x(), init.size_y())},
        epsilon{epsilon} {}

    // Constructor with rvalue matrix containing the boundary conditions in the outer
    // column and row and initial values in the inner matrix and given epsilon
    flow(Matrix&& init, T epsilon) noexcept : 
        values{Matrix(init), Matrix(std::move(init))}, 
        velocity{Matrix(init.size_x(), init.size_y()), Matrix(init.size_x(), init.size_y())},
        epsilon{epsilon} {}

    void jacobi(int);
    void derivative(int);
    
    void solve(int nthreads){
        jacobi(nthreads);
        derivative(nthreads);
    }

    T get_runtime() const
    {
        return runtime.count();
    }

    int save_and_plot(const std::string & filename) const;

    int save(const std::string & filename) const;

};

/*
    Solves the grid using jacobi iterations with openmp parallelization
*/
template<typename T>
inline void flow<T>::jacobi(int nthreads)
{

    int x_dim = (values[0]).size_x();
    int y_dim = (values[0]).size_y();

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    dim3 threadsPerBlockGrid;
    dim3 BlockGrid;

    threadsPerBlockGrid = dim3(32, 32);
    BlockGrid = dim3(16, numberOfSMs);

    T* device_matrix_1;
    T* device_matrix_2;
    T* diff;
    
    checkCuda(cudaMalloc(&device_matrix_1, x_dim * y_dim * sizeof(T)));
    checkCuda(cudaMalloc(&device_matrix_2, x_dim * y_dim * sizeof(T)));
    checkCuda(cudaMalloc(&diff, sizeof(T)));
    
    // Copy initial values from host to device
    checkCuda(cudaMemcpy(device_matrix_1, values[0].data(), x_dim * y_dim * sizeof(T), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(device_matrix_2, values[1].data(), x_dim * y_dim * sizeof(T), cudaMemcpyHostToDevice));
    
    // Launch Jacobi kernel
    cuda_jacobi<<<BlockGrid, threadsPerBlockGrid>>>(device_matrix_1, device_matrix_2, diff, epsilon, max_iter, x_dim, y_dim);
    
    checkCuda(cudaGetLastError());  
    checkCuda(cudaDeviceSynchronize()); 
    
    // Copy result back from device to host
    checkCuda(cudaMemcpy(values[0].data(), device_matrix_1, x_dim * y_dim * sizeof(T), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(values[1].data(), device_matrix_2, x_dim * y_dim * sizeof(T), cudaMemcpyDeviceToHost));
    
    checkCuda(cudaFree(device_matrix_1));
    checkCuda(cudaFree(device_matrix_2));

    for (int i = 0; i < x_dim; ++i) {
        for (int j = 0; j < y_dim; ++j) {
            std::cout <<  values[0](i,j) << " ";
        }
        std::cout << std::endl;
    }
  
/*
    T diff{};
    int iter{};


    auto start_time = std::chrono::high_resolution_clock::now();

    do
    {
        int old = iter % 2;
        int next = (iter + 1) % 2;

        diff = static_cast<T>(0);

        for (int i = 1; i < x_dim - 1; ++i) {
            for (int j = 1; j < y_dim - 1; ++j) {
    
                T tmp_value = 
                (values[old](i - 1, j) + values[old](i + 1, j) + 
                    values[old](i, j - 1) + values[old](i, j + 1)) * static_cast<T>(0.25);

                diff += (tmp_value - values[old](i, j)) * (tmp_value - values[old](i, j));
                values[next](i, j) = tmp_value;

            }
        }

        ++iter;
        result_id = next;

    }while (diff >= epsilon * epsilon && iter < max_iter);

    if(iter == max_iter) std::cerr << "maximum number of iterations reached, diff: " << diff << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    runtime = std::chrono::duration<double>(end_time - start_time);
*/
}

/*
    Computes the velocity for each point (x, y) using the final phi(x, y)  
    (excluding the boundary conditions)
*/
template<typename T>
inline void flow<T>::derivative(int nthreads)
{
    int x_dim = values[0].size_x();
    int y_dim = values[0].size_y();

    for (int i = 1; i < x_dim - 1; ++i) {
        for (int j = 1; j < y_dim - 1; ++j) {

            velocity[0](i, j) = 
            (values[result_id](i, j + 1) - values[result_id](i, j - 1)) * static_cast<T>(0.5);

            velocity[1](i, j) = 
            - (values[result_id](i + 1, j) - values[result_id](i - 1, j)) * static_cast<T>(0.5);

        }
    }

}

/*
    Saves the velocities to a file: "x y x_velocity y_velocity"
*/
template<typename T>
int flow<T>::save(const std::string & filename) const
{
    std::ofstream output_file;
    output_file.open(filename, std::ios::out | std::ios::binary );
    
    if(output_file.is_open()){

        for (int i = 1; i < values[0].size_x() - 1; ++i) {
            for (int j = 1; j < values[0].size_y() - 1; ++j) {
                T tmp_x_velocity = velocity[0](i, j);
                T tmp_y_velocity = velocity[1](i, j);

                output_file.write(reinterpret_cast<char*>(&tmp_x_velocity), sizeof(tmp_x_velocity));
                output_file.write(reinterpret_cast<char*>(&tmp_y_velocity), sizeof(tmp_y_velocity));
            }
        }
        output_file.close();
        return 0;
    }
    else{
        throw std::ios_base::failure("Unable to open file: " + filename);
    }
}

/*
    Saves the velocities to a file and plots the result
*/
template<typename T>
int flow<T>::save_and_plot(const std::string & filename) const
{
    save(filename);

    std::string x_size_str = std::to_string((values[0]).size_x() - 2);
    std::string y_size_str = std::to_string((values[0]).size_y() - 2);

    pid_t pid = fork();

    std::string precision;
    if constexpr (std::is_same_v<T, double>) {
        precision = "double";
    } 
    else if constexpr (std::is_same_v<T, float>) {
        precision = "single";
    }

    if (pid == 0) {  
        char* const args[] = {
            (char*)"../plot_env/bin/python3", 
            (char*)"../plot_vectorfield.py", 
            (char*)filename.c_str(), 
            (char*)x_size_str.c_str(), 
            (char*)y_size_str.c_str(), 
            (char*)precision.c_str(), NULL};

        if(execve("../plot_env/bin/python3", args, NULL) == -1){
            throw std::runtime_error("execve failed for plotting");
        }

    } 

    return 0;
}

#endif // FLOW_TYPE_H