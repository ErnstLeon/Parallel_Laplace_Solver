#ifndef FLOW_TYPE_MPI_H
#define FLOW_TYPE_MPI_H

#include <array>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <mpi.h>

#include <matrix_type.hpp>
#include <mpi_grid.hpp>

#ifndef MPI_TYPE
#define MPI_TYPE  MPI_FLOAT
#endif

/*
    Class containing routines for solving Laplace equation on a 2D grid with boundary conditions

    Template typename T : type used for calculation (double or float)

    Solves the Laplace equation ((d/dx)^2 + (d/dy)^2)phi(x,y) = 0
    using jacobi iterations, distributing the 2D grid over some processes
    and doing the jacobi iteration internally using mpi in combination with openmp. 

    Returns the x- and y-velocity for each point in a file (excluding the boundary conditions).
    File is binary with ordered like v_x(0, 0), v_y(0, 0), v_x(0, 1), v_y(0, 1) ...
*/
template<typename T>
class flow {

    using Matrix = matrix<T>;

private:

    // Two matrices, storing the (MPI-)local values of phi on the grid
    // containing old and new values during the iterative solution
    std::array<Matrix, 2> values{};

    // Two matrices to store the (MPI-)local velocities for each point on the grid;
    // first one to hold x-velocity, second to hold y-velocity
    std::array<Matrix, 2> velocity{};

    // Precision and max iteration count for iterative solution
    T epsilon{static_cast<T>(1e-8)};
    int max_iter = static_cast<int>(10000);

    // stores id of the value matrix that contains the final result 
    int result_id{};

    double runtime{};

    mpi_grid grid;

public:

    // Constructor with matrix containing the boundary conditions in the outer
    // column and row and initial values in the inner matrix
    flow(const Matrix& init) : 
    values{}, velocity{}, grid{init.size_x(), init.size_y()}
    {
        // resize the matrices to the shape of the (MPI-)local grid
        values[0].resize(grid.get_x_size(), grid.get_y_size());
        values[1].resize(grid.get_x_size(), grid.get_y_size());
        velocity[0].resize(grid.get_x_size(), grid.get_y_size());
        velocity[1].resize(grid.get_x_size(), grid.get_y_size());

        // copy values from inital matrix into the local matrix
        for(int i = 0; i < values[0].size_x(); ++i){
            {
                for(int j = 0; j < values[0].size_y(); ++j){
                    {   
                        auto tmp_value = init(grid.get_local_x_start() + i, grid.get_local_y_start() + j);
                        values[0](i, j) = tmp_value;
                        values[1](i, j) = tmp_value;
                    }
                }
            }
        }
    }

    // Constructor with rvalue matrix containing the boundary conditions in the outer
    // column and row and initial values in the inner matrix 
    flow(Matrix&& init) noexcept : 
    values{}, velocity{}, grid{init.size_x(), init.size_y()}
    {
        // resize the matrices to the shape of the (MPI-)local grid
        values[0].resize(grid.get_x_size(), grid.get_y_size());
        values[1].resize(grid.get_x_size(), grid.get_y_size());
        velocity[0].resize(grid.get_x_size(), grid.get_y_size());
        velocity[1].resize(grid.get_x_size(), grid.get_y_size());

        // copy values from inital matrix into the local matrix
        for(int i = 0; i < values[0].size_x(); ++i){
            {
                for(int j = 0; j < values[0].size_y(); ++j){
                    {   
                        auto tmp_value = init(grid.get_local_x_start() + i, grid.get_local_y_start() + j);
                        values[0](i, j) = tmp_value;
                        values[1](i, j) = tmp_value;
                    }
                }
            }
        }

        init.clear();
    }

    // Constructor with matrix containing the boundary conditions in the outer
    // column and row and initial values in the inner matrix and given max_iter
    flow(const Matrix& init, int max_iter) : 
    values{}, velocity{}, max_iter{max_iter}, grid{init.size_x(), init.size_y()}
    {
        // resize the matrices to the shape of the (MPI-)local grid
        values[0].resize(grid.get_x_size(), grid.get_y_size());
        values[1].resize(grid.get_x_size(), grid.get_y_size());
        velocity[0].resize(grid.get_x_size(), grid.get_y_size());
        velocity[1].resize(grid.get_x_size(), grid.get_y_size());

        // copy values from inital matrix into the local matrix
        for(int i = 0; i < values[0].size_x(); ++i){
            {
                for(int j = 0; j < values[0].size_y(); ++j){
                    {   
                        auto tmp_value = init(grid.get_local_x_start() + i, grid.get_local_y_start() + j);
                        values[0](i, j) = tmp_value;
                        values[1](i, j) = tmp_value;
                    }
                }
            }
        }
    }

    // Constructor with rvalue matrix containing the boundary conditions in the outer
    // column and row and initial values in the inner matrix and given max_iter
    flow(Matrix&& init, int max_iter) noexcept : 
    values{}, velocity{}, max_iter{max_iter}, grid{init.size_x(), init.size_y()}
    {
        // resize the matrices to the shape of the (MPI-)local grid
        values[0].resize(grid.get_x_size(), grid.get_y_size());
        values[1].resize(grid.get_x_size(), grid.get_y_size());
        velocity[0].resize(grid.get_x_size(), grid.get_y_size());
        velocity[1].resize(grid.get_x_size(), grid.get_y_size());

        // copy values from inital matrix into the local matrix
        for(int i = 0; i < values[0].size_x(); ++i){
            {
                for(int j = 0; j < values[0].size_y(); ++j){
                    {   
                        auto tmp_value = init(grid.get_local_x_start() + i, grid.get_local_y_start() + j);
                        values[0](i, j) = tmp_value;
                        values[1](i, j) = tmp_value;
                    }
                }
            }
        }

        init.clear();
    }

    void jacobi(int);
    void derivative(int);
    
    void solve(int nthreads){
        jacobi(nthreads);
        derivative(nthreads);
    }

    T get_runtime() const
    {
        return runtime;
    }

    int save_and_plot(const std::string & filename) const;

    int save(const std::string & filename) const;

    void cleanup() {
        grid.cleanup();
    }

};

/*
    Solves the local grid using jacobi iterations while exchanging the outer rows 
    and columns with the neighbouring processes using MPI send receive routines
*/
template<typename T>
inline void flow<T>::jacobi(int nthreads)
{
    T diff{};
    T global_diff{};
    int iter{};

    int x_dim = (values[0]).size_x();
    int y_dim = (values[0]).size_y();

    auto start_time = MPI_Wtime();

    // jacobi iteration until max iteration count is reached or 
    // difference between to iterations is small enough
    do
    {
        int old = iter % 2;
        int next = (iter + 1) % 2;

        diff = static_cast<T>(0);
        global_diff = static_cast<T>(0);

        // exchange of all four boundary with neighbouring processes
        MPI_Sendrecv(values[old].data(0, 1), 1, grid.get_column_type(), grid.get_previous_y_rank(), 0, 
            values[old].data(0, y_dim - 1), 1, grid.get_column_type(), grid.get_next_y_rank(), 0, grid.get_cart_comm(), MPI_STATUS_IGNORE);

        MPI_Sendrecv(values[old].data(0, y_dim - 2), 1, grid.get_column_type(), grid.get_next_y_rank(), 0, 
            values[old].data(0, 0), 1, grid.get_column_type(), grid.get_previous_y_rank(), 0, grid.get_cart_comm(), MPI_STATUS_IGNORE);

        MPI_Sendrecv(values[old].data(1, 0), 1, grid.get_row_type(), grid.get_previous_x_rank(), 0, 
            values[old].data(x_dim - 1 ,0), 1, grid.get_row_type(), grid.get_next_x_rank(), 0, grid.get_cart_comm(), MPI_STATUS_IGNORE);

        MPI_Sendrecv(values[old].data(x_dim - 2, 0), 1, grid.get_row_type(), grid.get_next_x_rank(), 0, 
            values[old].data(0, 0), 1, grid.get_row_type(), grid.get_previous_x_rank(), 0, grid.get_cart_comm(), MPI_STATUS_IGNORE);
    
        // one step of the jacobi iterations
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

        // sum the local difference to the previous iteration step
        MPI_Allreduce(&diff, &global_diff, 1, MPI_TYPE, MPI_SUM, grid.get_cart_comm());
        
        ++iter;
        result_id = next;

//    }while (global_diff >= epsilon * epsilon && iter < max_iter);
    }while (iter < max_iter);

    auto end_time = MPI_Wtime();
    runtime = (end_time - start_time);

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

    // updated the four boundary with neighbouring processes
    MPI_Sendrecv(values[result_id].data(0,1), 1, grid.get_column_type(), grid.get_previous_y_rank(), 0, 
        values[result_id].data(0,y_dim-1), 1, grid.get_column_type(), grid.get_next_y_rank(), 0, grid.get_cart_comm(), MPI_STATUS_IGNORE);

    MPI_Sendrecv(values[result_id].data(0,y_dim-2), 1, grid.get_column_type(), grid.get_next_y_rank(), 0, 
        values[result_id].data(0,0), 1, grid.get_column_type(), grid.get_previous_y_rank(), 0, grid.get_cart_comm(), MPI_STATUS_IGNORE);

    MPI_Sendrecv(values[result_id].data(1,0), 1, grid.get_row_type(), grid.get_previous_x_rank(), 0, 
        values[result_id].data(x_dim-1,0), 1, grid.get_row_type(), grid.get_next_x_rank(), 0, grid.get_cart_comm(), MPI_STATUS_IGNORE);

    MPI_Sendrecv(values[result_id].data(x_dim-2,0), 1, grid.get_row_type(), grid.get_next_x_rank(), 0, 
        values[result_id].data(0,0), 1, grid.get_row_type(), grid.get_previous_x_rank(), 0, grid.get_cart_comm(), MPI_STATUS_IGNORE);

    // compute the derivatives using simple finite differences 
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
    Saves the velocities to a file using MPI I/O routines: "x y x_velocity y_velocity"
*/
template<typename T>
int flow<T>::save(const std::string & filename) const
{
    MPI_File_delete(filename.c_str(), MPI_INFO_NULL);
    MPI_File output_file;

    int err = MPI_File_open(grid.get_cart_comm(), filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    
    if (err != MPI_SUCCESS) {
        throw std::ios_base::failure("Unable to open file: " + filename);
    }

    // create subarray datatype to give each process a file view of its individual sub matrix
    std::array<int, 2> array_size = {grid.get_global_x_size(), 2 * grid.get_global_y_size()};
    std::array<int, 2> sub_array_size = {grid.get_local_x_size(), 2 * grid.get_local_y_size()};
    std::array<int, 2> sub_array_start = {grid.get_local_x_start(), 2 * grid.get_local_y_start()};

    MPI_Datatype subarray;
    MPI_Type_create_subarray(2, array_size.data(), sub_array_size.data(), sub_array_start.data(), MPI_ORDER_C, MPI_TYPE, &subarray);
    MPI_Type_commit(&subarray);

    MPI_Offset offset = 0;
    MPI_File_set_view(output_file, offset, MPI_TYPE, subarray, "native", MPI_INFO_NULL);
    
    std::vector<T> buffer;
    for (int i = 1; i < values[0].size_x() - 1; ++i) {
        for (int j = 1; j < values[0].size_y() - 1; ++j) {
            buffer.push_back(velocity[0](i, j));
            buffer.push_back(velocity[1](i, j));
        }
    }

    // each process writes its induvidual submatrices to file
    MPI_File_write_all(output_file, buffer.data(), buffer.size(), MPI_TYPE, MPI_STATUS_IGNORE);

    MPI_File_close(&output_file);
    MPI_Type_free(&subarray);

    return 0;
}

/*
    Saves the velocities to a file and plots the result
*/
template<typename T>
int flow<T>::save_and_plot(const std::string & filename) const
{
    save(filename);

    std::string x_size_str = std::to_string(grid.get_global_x_size());
    std::string y_size_str = std::to_string(grid.get_global_y_size());

    if(grid.get_cart_comm_me() == 0){
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
                (char*)"/plot_env/bin/python3", 
                (char*)"/plot_vectorfield.py", 
                (char*)filename.c_str(), 
                (char*)x_size_str.c_str(), 
                (char*)y_size_str.c_str(), 
                (char*)precision.c_str(), NULL};

            if(execve("/plot_env/bin/python3", args, NULL) == -1){
                throw std::runtime_error("execve failed for plotting");
            }

        } 
    }

    return 0;
}

#endif // FLOW_TYPE_MPI_H