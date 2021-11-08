
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


def plot_2D_pos(data_arr):

    plt.figure()
    plt.plot(data[:,7], data[:,8])

def plot_Force(data_arr):

    plt.figure()

    plt.subplot(131)
    plt.plot(data_arr[:,1])
    plt.title("Force-x")

    plt.subplot(132)
    plt.plot(data_arr[:,2])
    plt.title("Force-y")

    plt.subplot(133)
    plt.plot(data_arr[:,3])
    plt.title("Force-z")

def plot_end_pos(data_arr):

    plt.figure()

    plt.subplot(131)
    plt.plot(data_arr[:,7])
    plt.title("x")

    plt.subplot(132)
    plt.plot(data_arr[:,8])
    plt.title("y")

    plt.subplot(133)
    plt.plot(data_arr[:,9])
    plt.title("z")

if __name__ == "__main__":

    data = np.load("processed_data/stand_still.npy")

    
    # plot_data(data, 9)
    # plot_data(data, 0)
    # visualize_states(data[2000:7000,:])
    visualize_states(data)
    plot_2D_pos(data)
    plot_Force(data)
    plot_end_pos(data)
    # plt.show()
    # data = np.load("processed_data/alternating.npy")
    # visualize_states(data[1000:3600,:])
    plt.show()