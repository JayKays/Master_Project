

import numpy as np
import matplotlib.pyplot as plt
import os
# from Data_preperation.visualize_data import pl

from Data_preperation.visualize_data import plot_states
from Data_preperation.filtering import lowpass_butter, median_filter

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

def cut_traj_starts(arr, cutoff_length = 100, traj_length = 1500):
    idx = cut_off_traj_starts_idx(arr.shape[0], cutoff_length, traj_length)
    return arr[idx,:]

def cut_off_traj_starts_idx(total_length, cutoff_length = 100, traj_length = 1500):
    num_traj = total_length//traj_length
    idx = np.arange(total_length)

    #Set indeces to be removed to -1
    for traj in range(num_traj):
        idx[traj*traj_length:traj*traj_length + cutoff_length] = -1

    return idx[idx != -1]

def filter_and_cut(data, trajectory_num, num_traj, windows_size = 3):
    new_traj_length = 1400

    states = data
    filtered_data = median_filter(states, window_size=windows_size)
    butter_filtered_data = lowpass_butter(filtered_data, cutoff=4)
    
    states = cut_traj_starts(states)[new_traj_length*trajectory_num:new_traj_length*(trajectory_num+num_traj),:]
    filtered_data = cut_traj_starts(filtered_data)[new_traj_length*trajectory_num:new_traj_length*(trajectory_num+num_traj),:]
    butter_filtered_data = cut_traj_starts(butter_filtered_data)[new_traj_length*trajectory_num:new_traj_length*(trajectory_num+num_traj),:]

    return states, filtered_data, butter_filtered_data


if __name__ == "__main__":

    # rb_dict = np.load("Old_data/Sine3_force_single/replay_buffer.npz")
    rb_dict = np.load("32_traj_varying_force/replay_buffer.npz")
    trajectory_num = 0
    num_traj = 3

    print("Num datapoints = ", rb_dict["obs"].shape[0])
    '''
    idx_list: 
    Force   0,  1,  2
    Torque  3,  4,  5
    Pos     6,  7,  8
    Vel     9, 10, 11
    Ang.vel 12, 13, 14
    '''
    states, median, butter = filter_and_cut(rb_dict["obs"], trajectory_num, num_traj,windows_size=5)
    states2, median2, butter2 = filter_and_cut(rb_dict["next_obs"],trajectory_num, num_traj, windows_size=5)
    states3, median3, butter3 = filter_and_cut(rb_dict["obs"] - rb_dict["next_obs"],trajectory_num, num_traj, windows_size=5)

    actions, act_med, act_but = filter_and_cut(rb_dict["action"], trajectory_num, num_traj, windows_size=5)
    
    print("Plotting states")
    states_to_plot = [0,1,2,6,7,8,9,10,11]
    states_to_plot = [2, 8, 11]
    states_to_plot = np.arange(rb_dict["obs"].shape[1])
    for state in []:
        # states_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        state_to_plot = [state]
        #

        plt.figure()
        plot_states(states, idx=state_to_plot)
        plot_states(median, idx=state_to_plot)
        plot_states(butter, idx=state_to_plot)

        plt.legend(["Median Filter", "Lowpass"])
        plt.savefig(f"BNN_data/states/state{state}")
        plt.show()
        plt.close()

        plt.figure()
        plot_states(butter - butter2, idx=state_to_plot)

        plt.legend(["Raw Data", "Median Filter"])
        plt.show()
        # plt.savefig(f"BNN_data/state_diff/state{state}")
    
    plt.figure()
    plot_states(states, idx=[2])
    plot_states(median, idx=[2])
    plt.legend(["Median Filter", "Lowpass"])
    plt.ylabel("Force [N]")
    plt.xlabel("Time [s]")
    plt.savefig("figures_for_report/median_lowpass.eps", format = "eps")
    plt.show()
    plt.close()

    plt.figure()
    plot_states(states, idx=[11])
    plot_states(median, idx=[11])
    plt.legend(["Raw Data", "Median Filter"])
    plt.ylabel("Velocity [m/s]")
    plt.xlabel("Time [s]")
    plt.savefig("figures_for_report/raw_median.eps", format = "eps")
    plt.show()

    # plt.figure()
    print("Plotting Actions")
    print(actions.shape)
    for act in range(actions.shape[1]):
        plt.figure()
        plot_states(actions, idx=[act])
        plot_states(act_med, idx=[act])
        plot_states(act_but, idx=[act])
        plt.legend(["Raw data", "Median Filter", "Lowpass"])
        plt.savefig(f"BNN_data//actions/action{act}")
        plt.close()

