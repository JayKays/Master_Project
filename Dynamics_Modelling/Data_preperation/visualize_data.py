
from matplotlib import pyplot as plt
import numpy as np


# STATES = ["time", "Force-x", "Force-y", "Force-z", \
#     "Torque-x", "Torque-y", "Torque-z", "Pos-x", "Pos-y", \
#         "Pos-z", "Vel-x", "Vel-y", "Vel-z", "Angular-x", "Angular-y", "Angular-z"]

STATES = ["Force-x", "Force-y", "Force-z", \
    "Torque-x", "Torque-y", "Torque-z", "Pos-x", "Pos-y", \
        "Pos-z", "Vel-x", "Vel-y", "Vel-z", "Angular-x", "Angular-y", "Angular-z"]
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

def plot_Force(data_arr, force_idx = [1,2,3]):

    plt.figure()

    plt.subplot(131)
    plt.plot(data_arr[:,force_idx[0]])
    plt.title("Force-x")

    plt.subplot(132)
    plt.plot(data_arr[:,force_idx[1]])
    plt.title("Force-y")

    plt.subplot(133)
    plt.plot(data_arr[:,force_idx[2]])
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

def plot_filter_comp(data_arr, filtered_data, idx, f_idx = None):

    if f_idx is None:
        f_idx = idx
    num_plots = len(idx)

    plt.figure()

    for n in range(num_plots):
        plt.subplot(num_plots, 1, n+1)
        plt.plot(data_arr[:,idx[n]])
        if idx[n] in f_idx:
            plt.plot(filtered_data[:,idx[n]])
            plt.legend(["data", "filtered"])
        plt.title(STATES[idx[n]])

def plot_states(data_arr, idx):

    num_plots = len(idx)

    # plt.figure()
    x = np.arange(data_arr.shape[0])/100
    for n in range(num_plots):
        plt.subplot(num_plots,1, n+1)
        plt.plot(x,data_arr[:,idx[n]])
        plt.title(STATES[idx[n]])

    # plt.legend(["data", "filtered"])


if __name__ == "__main__":
    import numpy as np
    from filtering import lowpass_fft, lowpass_butter

    data_arr = np.load("Old_data/timestep_test2/timesteps.npy")

    print(np.median(data_arr))
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(0.65 * data_arr[1:])
    plt.xlabel("Time step")
    plt.ylabel("Step duration [s]")
    plt.title("Time step lengths")
    plt.subplot(122)
    plt.hist(0.65*data_arr[1:])
    plt.title("Time step length distribution")
    plt.ylabel("Time step count")
    plt.xlabel("Step duration [s]")
    # plot_filter_comp(data_arr[:,:], lowpass_butter(data_arr.copy(), fs = 100, cutoff = 4, idx = [3, 9, 12])[:,:], idx=[3,9,12], f_idx=[3,9,12])
    plt.savefig("figures_for_report/timesteps.eps", format="eps")
    plt.show()

    '''
    idx_list: 
    t       0
    Force   1,  2,  3
    Torque  4,  5,  6
    Pos     7,  8,  9
    Vel     10, 11, 12
    Ang.vel 13, 14, 15
    '''
