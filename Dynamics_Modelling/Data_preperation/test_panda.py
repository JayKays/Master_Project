import gym
from franka_interface import ArmInterface
from panda_robot import PandaArm
#from gym import spaces
import numpy as np
import rospy
from gym import logger, spaces
from gym.utils import seeding
import quaternion
import sys
sys.path.append("/home/jensek/Project/src/Compliant_control/gym-panda/HFMC")
# from gym_panda.envs import HFMC_func as func
#from gym_panda.envs.HFMC_ObsSpace import ObservationSpace
# from gym_panda.envs import HFMC_config as cfg
sim = True
rospy.init_node("HFMC")
# rate = rospy.Rate(cfg.PUBLISH_RATE)
robot = ArmInterface()
# robot = PandaArm()
joint_names = robot.joint_names()
#print(joint_names)
state_list = np.zeros((15, 100000))
rate = rospy.Rate(1000)

# mvt = robot.get_movegroup_interface()

# for i in range (10000):
#     state_dict = (robot.get_full_state_space())
#     state_list[:, i] = [state_dict["Fx"], state_dict["Fy"], state_dict["Fz"] , state_dict["Tx"], \
#              state_dict["Ty"], state_dict["Tz"], state_dict["x"], \
#              state_dict["y"] , state_dict["z"] , state_dict["Vx"], \
#              state_dict["Vy"], state_dict["Vz"], state_dict["Ax"], state_dict["Ay"],
#              state_dict["Az"]]
#     rate.sleep()

# print(state_list)

[-0.18691758799142744, -0.3217289191503339, 1.8054475831171266,
-0.3155675455713534, 0.2768411430532767, 0.025608736411,
0.306913056192768, 9.13114345996483e-06, 0.5901851533453434,
0.001044509697293404, 0.008710911905803333, 0.007507693845972836,
-0.006299526318762417, 0.011584745159985999, 0.008745220138213234]

robot.move_to_neutral()

# # for i in range (15):
# #     print ("index: ", i, "  min: ", np.min(state_list[i,:]),  "  max: ", np.max(state_list[i,:]))


def get_z_limits():
    # state_dict = (robot.get_full_state_space())
    # state_list = [state_dict["Fx"], state_dict["Fy"], state_dict["Fz"] , state_dict["Tx"], \
    #             state_dict["Ty"], state_dict["Tz"], state_dict["x"], \
    #             state_dict["y"] , state_dict["z"] , state_dict["Vx"], \
    #             state_dict["Vy"], state_dict["Vz"], state_dict["Ax"], state_dict["Ay"],
    #             state_dict["Az"]]


    # pos = state_list[6:9]
    # orientation = robot.endpoint_pose()["orientation"]
    # robot.move_to_cartesian_pose(pos, ori=orientation)

    pos, ori = init_robot()

    z_min = pos[-1]
    z_max = z_min

    best_min = 10
    best_max = 10

    f_min = 3
    f_max = 10

    for i in range(20):
        state_dict = robot.get_full_state_space()
        
        # print("Force in z-direction: ", state_dict["Fz"])
        # print("End defector z pos:   ", state_dict["z"])

        pos[-1] += (-1) * 0.0005

        if abs(f_min - state_dict["Fz"]) < best_min:
            z_min = state_dict["z"]
            best_min = abs(f_min - state_dict["Fz"])
            print("Found closest force to ", f_min, ": ", state_dict["Fz"])

        if abs(f_max - state_dict["Fz"]) < best_max:
            z_max = state_dict["z"]
            best_max = abs(f_max - state_dict["Fz"])
            print("Found closest force to ", f_max, ": ", state_dict["Fz"])

        robot.move_to_cartesian_pose(pos, ori=ori)

    reset_robot()
    print("Z limits: ", z_min, "--", z_max)
    print("Force min max: ", best_min, best_max)  

def reset_robot():
    robot.move_to_neutral()


def init_robot():
    reset_robot()

    state_dict = (robot.get_full_state_space())
    state_list = [state_dict["Fx"], state_dict["Fy"], state_dict["Fz"] , state_dict["Tx"], \
                state_dict["Ty"], state_dict["Tz"], state_dict["x"], \
                state_dict["y"] , state_dict["z"] , state_dict["Vx"], \
                state_dict["Vy"], state_dict["Vz"], state_dict["Ax"], state_dict["Ay"],
                state_dict["Az"]]

    pos = state_list[6:9]
    orientation = robot.endpoint_pose()["orientation"]
    robot.move_to_cartesian_pose(pos, ori=orientation)


    return pos, orientation


def gather_data(z_lims, init_pos, init_ori, num_samples = 10):

    pos = init_pos
    ori = init_ori
    state_list = np.zeros((15,num_samples))

    for i in range(num_samples):
        pos[-1] = np.random.uniform(z_lims[0], z_lims[1])

        robot.move_to_cartesian_pose(pos, ori=ori)

        state_dict = (robot.get_full_state_space())
        state_list[:,i]= [state_dict["Fx"], state_dict["Fy"], state_dict["Fz"] , state_dict["Tx"], \
                    state_dict["Ty"], state_dict["Tz"], state_dict["x"], \
                    state_dict["y"] , state_dict["z"] , state_dict["Vx"], \
                    state_dict["Vy"], state_dict["Vz"], state_dict["Ax"], state_dict["Ay"],
                    state_dict["Az"]]

    return state_list

def print_state(state):

    print(robot.get_full_state_space()[state])

def move_z(z, pos = None, ori = None):

    if pos is None or ori is None:
        pos, ori = init_robot()

    pos[-1] = z

    robot.move_to_cartesian_pose(pos, ori)

    print_state('z')


if __name__ == "__main__":

    max_force_z = 0.5819466433511261
    temp_z = 0.5781
    
    # z_limits = [0.5819352100006313, 0.5854802564152597]
    z_limits = [0.58193, 0.58548]
    # get_z_limits()
    pos, ori = init_robot()
    print(pos)
    # move_z(max_force_z, pos, ori)

    print_state('Vz')
    # print(gather_data(z_limits, pos, ori)[2,:])

    # reset_robot()
    # for i in range(10000):
    #     print_state("Fx")







