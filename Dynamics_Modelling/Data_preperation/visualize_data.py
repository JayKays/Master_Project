
from matplotlib import pyplot as plt

import numpy as np





def plot_data(data_arr, idx):
    plt.figure()
    plt.plot(data_arr[:,idx])
    # plt.show()

def hist_data(data_arr, idx):
    plt.figure
    plt.hist(data_arr[:,idx])
    # plt.show()

def visualize_states(data_arr):

    plt.figure()

    plt.subplot(221)
    plt.plot(data_arr[:,0])
    plt.title("timestep")

    plt.subplot(222)
    plt.plot(data_arr[:,9])
    plt.title("z-pos")

    plt.subplot(223)
    plt.hist(data_arr[:,3])
    plt.title("z-Force")

    plt.subplot(224)
    plt.plot(data_arr[:,12])
    plt.title("z-Vel")

if __name__ == "__main__":

    data = np.load("processed_data/alternating.npy")

    
    # plot_data(data, 9)
    # plot_data(data, 0)
    # visualize_states(data[2000:7000,:])
    visualize_states(data)
    plt.show()
    visualize_states(data[1:3700,:])
    plt.show()