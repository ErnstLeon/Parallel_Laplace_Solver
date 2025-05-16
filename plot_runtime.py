import matplotlib.pyplot as plt
import numpy as np

# Load the data
omp = np.loadtxt('runtime_omp.dat')
mpi = np.loadtxt('runtime_mpi.dat')
cuda = np.loadtxt('runtime_cuda.dat')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(omp[:, 0], omp[:, 1], marker='o', label='OpenMP - 6 Threads')
plt.plot(mpi[:, 0], mpi[:, 1], marker='s', label='MPI - 6 Processes / 6 Threads')
plt.plot(cuda[:, 0], cuda[:, 1], marker='^', label='CUDA - Gforce RTX2060')

# Formatting
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Grid size')
plt.ylabel('Runtime (s)')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig('performance_plot.pdf')