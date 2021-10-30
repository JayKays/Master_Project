import gym
import gym_panda
import random
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

 

if __name__ == "__main__":
    print('started')
    env = gym.make('panda-VIC-v0')
    #agent = Agent(env.action_space)
    number_of_runs = 1
    print("action space: ", env.action_space)
    print(env.observation_space)
