import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_runtime(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            threads, runtime = map(float, line.split())
            if threads not in data:
                data[threads] = []
            data[threads].append(runtime)

    # Prepare data for plotting
    threads = []
    mean_runtimes = []
    errors = []

    for thread_count, runtimes in data.items():
        threads.append(thread_count)
        mean_runtimes.append(np.mean(runtimes))
        std_dev = np.std(runtimes)  # Standard deviation
        n = len(runtimes)  # Number of data points (should be 10)
        sem = std_dev / np.sqrt(n)  # Standard error of the mean
        errors.append(sem)

    # Convert to numpy arrays for plotting
    threads = np.array(threads)
    mean_runtimes = np.array(mean_runtimes)
    errors = np.array(errors)

    # Plot with error bars
    plt.errorbar(threads, mean_runtimes, yerr=errors, fmt='o', capsize=5, label='Runtime', color='blue')

    # Add labels and title
    plt.xlabel('Number of Threads')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Number of Threads')

    # Show the legend
    plt.legend()

    # Sace the plot
    plt.savefig("./omp_runtime.pdf")

if __name__ == "__main__":

    filename = sys.argv[1]
    plot_runtime(filename)