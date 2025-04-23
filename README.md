# Jacobi Iteration

This project implements the **Jacobi iteration** method to solve the **Laplace equation**.

Given an incompressible and irrotational fluid, the stream function $\phi(x, y)$ must satisfy the Laplace equation:

```math
\frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} = 0 \ .
````

The velocity components of the flow are given by:

```math
v_x = \frac{\partial \phi}{\partial y}, \ \ v_y = -\frac{\partial \phi}{\partial x} \ .
```

Using finite differences, the Laplace equation can be solved iteratively based on an initial matrix storing $\phi(x, y)_{\text{init}}$.  
The boundary values of this matrix define the boundary conditions of the flow, such as the velocity of the incoming and outgoing flow in a cavity. The resulting solution represents the flow behavior inside the cavity.

## How to Use

The input matrix must be of type `matrix<T>`, which is based on `std::vector<T>`. You can initialize it with a default value for each entry like this:

```cpp
matrix<T> init{x_dim, y_dim, static_cast<T>(0)};
````
- `x_dim` and `y_dim` specify the dimensions of the matrix.
- The third argument sets the default value for all entries during initialization.

This initial matrix (especially the boundary conditions) defines a flow. This flow is initialized like:
```cpp
flow<T> psi {std::move(init), max_iter};
````
- `max_iter` specifies the number of iterations used to solve the laplace equation.

To solve the equation and save the resulting velocities, use:
```cpp
psi.solve(nthreads);
psi.save(output_filename);
````
The velocities are saved in binary form like to the file, with the following layout per line: x y x_velocity y_velocity.

## Parallelization
Headers for CUDA, MPI+OpenMP, and OpenMP versions are available.

Each header contains a specific implementation of the `flow` class and solves the Laplace equation in parallel, based on the available system.

## Example 

An example of computing the flow for a certain initial matrix is provided in main.cpp (MPI+OMP) and main.cu(CUDA).