import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import sys
import os

def read_binary_velocity_field(filename, global_x_size, global_y_size, precision):
    
    if precision == "single":
        data = np.fromfile(filename, dtype=np.float32)
    elif precision == "double":
        data = np.fromfile(filename, dtype=np.float64)
    else:
        data = np.fromfile(filename, dtype=np.float32)

    expected_size = 2 * global_x_size * global_y_size
    if data.size != expected_size:
        raise ValueError(f"Unexpected file size! Expected {expected_size} values, got {data.size}.")
    
    x = []
    y = []
    x_velocity = []
    y_velocity = []

    for i in range(global_x_size):
        for j in range(global_y_size):
            x.append(i + 1)
            y.append(j + 1)
            x_velocity.append(data[global_y_size * 2 * i + 2 * j])
            y_velocity.append(data[global_y_size * 2 * i + 2 * j + 1])

    x = np.array(x)
    y = np.array(y)
    x_velocity = np.array(x_velocity)
    y_velocity = np.array(y_velocity)

    magnitudes = np.sqrt(x_velocity**2 + y_velocity**2)
    non_zero_magnitudes = magnitudes != 0

    x_velocity[non_zero_magnitudes] /= magnitudes[non_zero_magnitudes]
    y_velocity[non_zero_magnitudes] /= magnitudes[non_zero_magnitudes]

    return x, y, x_velocity, y_velocity, magnitudes


def plot_velocity_field(filename, global_x_size, global_y_size, precision):
    
    x, y, x_velocity, y_velocity, magnitudes = read_binary_velocity_field(filename, global_x_size, global_y_size, precision)

    fig, ax = plt.subplots()

    x_unique = np.unique(x)[::2]
    y_unique = np.unique(y)[::2]

    mask = np.isin(x, x_unique) & np.isin(y, y_unique)

    x_ = x[mask]
    y_ = y[mask]
    x_velocity_ = x_velocity[mask]
    y_velocity_ = y_velocity[mask]

    ax.quiver(x_, y_, x_velocity_, y_velocity_, scale=50)


    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    Z = magnitudes.reshape(X.shape, order='F')

    c = ax.imshow(Z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], origin='lower', cmap='autumn', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Vector Field')

    fig.colorbar(c, ax=ax)

    outputname = os.path.splitext(filename)[0] + ".png"
    plt.savefig(outputname)


if __name__ == "__main__":

    binary_filename = sys.argv[1]
    global_x_size = int(sys.argv[2])
    global_y_size = int(sys.argv[3])
    precision = str(sys.argv[4])

    plot_velocity_field(binary_filename, global_x_size - 2, global_y_size - 2, precision)