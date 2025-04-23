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

/*
    Class containing routines for solving Laplace equation on a 2D grid with boundary conditions

    Template typename T : type used for calculation (double or float)

    Solves the Laplace equation ((d/dx)^2 + (d/dy)^2)phi(x,y) = 0
    using jacobi iterations with openmp. 

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
    // column and row and initial values in the inner matrix and given max_iter
    flow(const Matrix& init, int max_iter) : 
        values{Matrix(init), Matrix(init)},
        velocity{Matrix(init.size_x(), init.size_y()), Matrix(init.size_x(), init.size_y())},
        max_iter{max_iter} {}

    // Constructor with rvalue matrix containing the boundary conditions in the outer
    // column and row and initial values in the inner matrix and given max_iter
    flow(Matrix&& init, int max_iter) noexcept : 
        values{Matrix(init), Matrix(std::move(init))}, 
        velocity{Matrix(init.size_x(), init.size_y()), Matrix(init.size_x(), init.size_y())},
        max_iter{max_iter} {}

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
    T diff{};
    int iter{};

    int x_dim = (values[0]).size_x();
    int y_dim = (values[0]).size_y();

    auto start_time = std::chrono::high_resolution_clock::now();

    do
    {
        int old = iter % 2;
        int next = (iter + 1) % 2;

        diff = static_cast<T>(0);

        #pragma omp parallel for shared(values, old, next) reduction(+ : diff) num_threads(nthreads)
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

//    }while (diff >= epsilon * epsilon && iter < max_iter);
    }while (iter < max_iter);

    auto end_time = std::chrono::high_resolution_clock::now();
    runtime = std::chrono::duration<double>(end_time - start_time);

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

    #pragma omp parallel for shared(values, velocity) num_threads(nthreads)
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