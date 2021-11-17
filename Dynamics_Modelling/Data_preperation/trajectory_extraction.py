
from visualize_data import plot_states
from filtering import lowpass_fft
import numpy as np
import matplotlib.pyplot as plt




'''
idx_list: 
t       0
Force   1,  2,  3
Torque  4,  5,  6
Pos     7,  8,  9
Vel     10, 11, 12
Ang.vel 13, 14, 15
'''


def load_data(filename):
    if not filename.endswith(".npy"):
        filename += ".npy"

    return np.load(f"processed_data/{filename}")


if __name__ == "__main__":

    data = load_data("20_traj_data")

    np.save("training_data/single_traj.npy", data[21730:23600,:])
    plot_states(data[21730:23600,:], [3,9, 12])
    plt.show()