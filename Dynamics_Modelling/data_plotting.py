

import numpy as np
import matplotlib.pyplot as plt
import os
# from Data_preperation.visualize_data import pl

def plot_training_data(log_dir):
    plot_dir = log_dir + "/training_data"
    os.makedirs(plot_dir, exist_ok=True)

    data = np.load(log_dir + "/replay_buffer.npz")

    state_observations= data["obs"]

    for i in range(state_observations.shape[1]):
        plt.figure()
        plt.plot(state_observations[:,i])
        plt.title(f"Observations dim {i}")
        plt.savefig(plot_dir + f"/obs_dim{i}.png")
        plt.close()


def plot_train_loss(log_dir):
    plot_dir = log_dir + "/loss"
    os.makedirs(plot_dir, exist_ok=True)

    