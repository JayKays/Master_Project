

import sys
sys.path.append("/home/jaykay/Master_Project/src/Compliant_control/gym-panda/VIC")
from gym_panda.envs import VIC_func as func
from gym_panda.envs import VIC_config as cfg

from matplotlib import pyplot as plt
import numpy as np

def plot_force_traj(num_traj = 10):
    num_itr = cfg.duration * cfg.PUBLISH_RATE
    T = 1/cfg.PUBLISH_RATE
    plt.figure()

    for _ in range(num_traj):
        s = func.generate_Fd_random(num_itr, cfg.Fd, T, slope_prob = .6)
        plt.plot(s[2,:])
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    plot_force_traj()