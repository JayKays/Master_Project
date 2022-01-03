

from Data_preperation.Panda_VIC import VIC_Env
import rospy
import sys
import numpy as np
import time

from mbrl.util.replay_buffer import ReplayBuffer
from util import seed_everything

#Needs to be set to appropriate local path
sys.path.append("/home/jaykay/Master_Project/src/Compliant_control/gym-panda/VIC")
from gym_panda.envs import VIC_func as func
from gym_panda.envs import VIC_config as VIC_cfg

'''
Note:

This requires the gazebo simulator and planning to be running before execution

'''
def sim_and_record(num_trials, log_dir, debug_mode = True, state_action_size = (15,6), record_timesteps = False, constant_force = False):
    '''
    Runs the robot simulation and records state and action data to a replayBuffer

    Paremeters:

    '''
    sim = True
    rospy.init_node("VIC")
    env = VIC_Env()

    replay_buffer = ReplayBuffer(
        env.max_num_it*num_trials,
        (state_action_size[0],),
        (state_action_size[1],),
        max_trajectory_length=env.max_num_it
    )

    if constant_force:
        force = func.generate_Fd_constant(env.max_num_it, VIC_cfg.Fd)
        env.set_fd_constant(force)
        print("Desired force set to constant:\n", env.f_d)

    timesteps = np.zeros(replay_buffer.capacity)

    gamma_mean = np.array([0.001 , 0.0001])
    gamma_std = gamma_mean / 5
    env.set_not_learning()
    gammas = []
    env_steps = 0
    current_trial = 0
    
    while env_steps < replay_buffer.capacity:
        # action_gamma = np.random.uniform(VIC_cfg.GAMMA_B_LOWER, VIC_cfg.GAMMA_B_UPPER, 2)
        action_gamma = np.random.normal(gamma_mean, gamma_std)
        gammas.append(action_gamma)
        print("Trial gamma:",action_gamma)
        obs = env.reset()
        obs = state_list_from_dict(env.robot.get_full_state_space())
        done = False
        total_reward = 0.0
        steps_trial = 0
        start = time.time()
        while not done:
            _, reward, done, _ = env.step(action_gamma)

            action = env.get_wrench()
            
            next_obs = state_list_from_dict(env.robot.get_full_state_space())
            end  = time.time()
            
            replay_buffer.add(obs, action[:,0], next_obs, reward, done)
            timesteps[env_steps] = end - start
            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1
            start = end
            
            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")
        
        if log_dir is not None:
            replay_buffer.save(log_dir)
            if record_timesteps:
                np.save(log_dir + "/timesteps.npy", timesteps)

        current_trial += 1
        if debug_mode:
            print(f"Trial: {current_trial }, reward: {total_reward}.")


    print(f"Finished {env_steps} over {current_trial} trials")
    if debug_mode:
        for gamma in gammas:
            print(f"Gamma values: {gamma[0]}, {gamma[1]}")

def state_list_from_dict(state_dict):
    state_list = np.array([state_dict["Fx"], state_dict["Fy"], state_dict["Fz"] , state_dict["Tx"], \
                state_dict["Ty"], state_dict["Tz"], state_dict["x"], \
                state_dict["y"] , state_dict["z"] , state_dict["Vx"], \
                state_dict["Vy"], state_dict["Vz"], state_dict["Ax"], state_dict["Ay"],
                state_dict["Az"]])
    return state_list

if __name__ == "__main__":
    import os

    seed_everything(42)

    log_dir = "5_traj_const_force_rand_g"
    log_dir = "32_traj_varying_force"
    log_dir = "32_traj_constant_force"
    log_dir = "timestep_test2"
    # os.makedirs(log_dir)
    sim_and_record(1, log_dir = None, constant_force=False)

    