#ifndef MPI_GRID_H
#define MPI_GRID_H

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <mpi.h>

#ifndef MPI_TYPE
#define MPI_TYPE  MPI_FLOAT
#endif

/*
    Class containing objects for doing 2D domain decomposition using MPI

    A 2D grid of size "global_x_size" x "global_y_size" is split up in smaller blocks 
    and distributed over MPI processes. A MPI cartesian communicator is used and the processes 
    are distributed over the two dimensions using MPI routines. Every local grid containes two additional
    rows and columns to store values neighbouring the local grid for a jacobi iteration.
*/
class mpi_grid{
private:

    // process id
    int mpi_me {}; 
    // number of overall MPI processes
    int mpi_nps {};

    // process ids of neighbouring processes in both
    // dimensions
    int previous_x_rank {};
    int next_x_rank {};
    
    int previous_y_rank {};
    int next_y_rank {};

    //extend of the MPI cartesian grid and the local coordinated within
    std::array<int, 2> mpi_cart_dimensions {};
    std::array<int, 2> mpi_cart_coordinates {};

    //Communicator and datatypes for IPC
    MPI_Comm mpi_cart_com;
    MPI_Datatype mpi_column;
    MPI_Datatype mpi_row;

    // overall size of the global grid without 
    // boundaries
    int global_inner_x_size{};
    int global_inner_y_size{};

    // starting x coordinate and range of 
    // the local grid in the overall grid
    int local_x_start{};
    int local_x_size{};

    // starting y coordinate and range of 
    // the local grid in the overall grid
    int local_y_start{};
    int local_y_size{};

    // overall size of the local grid with overlapp 
    // for neighbouring values
    int x_size{};
    int y_size{};

public:

    void get_local_extent(int global_x_size, int global_y_size);

    void initiate_mpi_grid();

    // Constructor with initial values
    mpi_grid(int global_x_size, int global_y_size)
    {
        //initiate MPI grid 
        initiate_mpi_grid();

        //get local grid and extends
        get_local_extent(global_x_size, global_y_size);

        //generate MPI datatypes for IPC
        MPI_Type_vector(x_size, 1, y_size, MPI_TYPE, &mpi_column);
        MPI_Type_contiguous(y_size, MPI_TYPE, &mpi_row);

        MPI_Type_commit(&mpi_column);
        MPI_Type_commit(&mpi_row);
    }

    void cleanup() {
        MPI_Type_free(&mpi_column);
        MPI_Type_free(&mpi_row);
        MPI_Comm_free(&mpi_cart_com);
    }

    MPI_Datatype get_column_type() const {
        return mpi_column;
    }

    MPI_Datatype get_row_type() const {
        return mpi_row;
    }

    MPI_Comm get_cart_comm() const {
        return mpi_cart_com;
    }

    const std::array<int, 2>& get_cart_comm_dimensions() const {
        return mpi_cart_dimensions;
    }

    const std::array<int, 2>& get_cart_comm_coordinates() const {
        return mpi_cart_coordinates;
    }

    int get_cart_comm_me() const {
        return mpi_me;
    }

    int get_x_size() const {
        return x_size;
    }

    int get_y_size() const {
        return y_size;
    }

    int get_local_x_start() const {
        return local_x_start;
    }

    int get_local_x_size() const {
        return local_x_size;
    }

    int get_local_y_start() const {
        return local_y_start;
    }

    int get_local_y_size() const {
        return local_y_size;
    }

    int get_global_x_size() const {
        return global_inner_x_size;
    }

    int get_global_y_size() const {
        return global_inner_y_size;
    }

    int get_next_x_rank() const {
        return next_x_rank;
    }

    int get_next_y_rank() const {
        return next_y_rank;
    }

    int get_previous_x_rank() const {
        return previous_x_rank;
    }

    int get_previous_y_rank() const {
        return previous_y_rank;
    }
};

/*
    Distributes global 2D grid over all MPI processes and computes the extent of the local grid
*/
inline void mpi_grid::get_local_extent(int global_x_size, int global_y_size){

    // distibutes the global grid without the boundary conditions,
    // i.e. without the outer columns and rows
    global_inner_x_size = global_x_size - static_cast<int>(2);
    global_inner_y_size = global_y_size - static_cast<int>(2);

    // first the grid is evenly distributed
    local_x_size = global_inner_x_size / mpi_cart_dimensions[0];
    local_y_size = global_inner_y_size / mpi_cart_dimensions[1];
    
    // remains are evenly distributed, one to each process;
    // given the size of each local grid, the starting point for 
    // each local grid in the global grid in calculated
    if(static_cast<int>(mpi_cart_coordinates[0]) < (global_inner_x_size % static_cast<int>(mpi_cart_dimensions[0]))){
        local_x_size += static_cast<int>(1);
        local_x_start = mpi_cart_coordinates[0] * local_x_size;
    }
    else{
        local_x_start = mpi_cart_coordinates[0] * local_x_size + global_inner_x_size % static_cast<int>(mpi_cart_dimensions[0]);
    }
            
    if(static_cast<int>(mpi_cart_coordinates[1]) < (global_inner_y_size % static_cast<int>(mpi_cart_dimensions[1]))){
        local_y_size += static_cast<int>(1);
        local_y_start = mpi_cart_coordinates[1] * local_y_size;
    }
    else{
        local_y_start = mpi_cart_coordinates[1] * local_y_size + global_inner_y_size % static_cast<int>(mpi_cart_dimensions[1]);
    }

    // overall size of local grid containes 2 more rows and columns for neighbouring values
    x_size = local_x_size + static_cast<int>(2);
    y_size = local_y_size + static_cast<int>(2);
}

/*
    Initialtes MPI cartesian communicator 
*/
inline void mpi_grid::initiate_mpi_grid(){  

    // get number of MPI processes and current rank
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_nps);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_me);

    std::array<int, 2> periods = {0, 0};

    //create cartesian communicator
    MPI_Dims_create(mpi_nps, 2, mpi_cart_dimensions.data());
    MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_cart_dimensions.data(),
        periods.data(), false, &mpi_cart_com);
    MPI_Cart_coords(mpi_cart_com, mpi_me, 2, mpi_cart_coordinates.data());

    // get ranks of neighbouring processes in 2D grid
    MPI_Cart_shift(mpi_cart_com, 0, 1, &previous_x_rank, &next_x_rank);
    MPI_Cart_shift(mpi_cart_com, 1, 1, &previous_y_rank, &next_y_rank);
        
}

#endif // MPI_GRID_H