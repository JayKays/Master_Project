
from franka_interface import ArmInterface
import numpy as np
import rospy
import sys
from time import sleep


sys.path.append("/home/jensek/Project/src/Compliant_control/gym-panda/HFMC")
sim = True
rospy.init_node("HFMC")
robot = ArmInterface()
rate = rospy.Rate(1000)
robot.move_to_neutral()

# [-0.18691758799142744, -0.3217289191503339, 1.8054475831171266,
# -0.3155675455713534, 0.2768411430532767, 0.025608736411,
# 0.306913056192768, 9.13114345996483e-06, 0.5901851533453434,
# 0.001044509697293404, 0.008710911905803333, 0.007507693845972836,
# -0.006299526318762417, 0.011584745159985999, 0.008745220138213234]

def reset_robot():
    robot.move_to_neutral()

def get_z_limits(f_lims = np.array([3,10])):

    pos, ori = init_robot()

    z_min = pos[-1]
    z_max = z_min

    best_min = 10
    best_max = 10

    f_min = f_lims[0]
    f_max = f_lims[-1]

    for i in range(20):
        state_dict = robot.get_full_state_space()

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



def print_state(state):
    print(robot.get_full_state_space()[state])

def move_z(z, relative = True, pos = None, ori = None):

    if pos is None or ori is None:
        pos, ori = init_robot()

    if relative:
        pos[-1] += z
    else:
        pos[-1] = z

    robot.move_to_cartesian_pose(pos, ori)

    return pos, ori

def alternating_motion_z(z_lims, init_pos, init_ori, num_samples = 100):

    pos = init_pos
    ori = init_ori
    state_list = np.zeros((15,num_samples))

    for i in range(num_samples):
        z = z_limits[i%2]

        pos, ori = move_z(z, relative=False, pos = pos, ori=ori)

def random_motion_z(z_lims, init_pos, init_ori, num_samples = 100):

    pos = init_pos
    ori = init_ori
    state_list = np.zeros((15,num_samples))

    for i in range(num_samples):
        z = np.random.uniform(z_lims[0], z_lims[-1])

        pos, ori = move_z(z, relative=False, pos = pos, ori=ori)

if __name__ == "__main__":

    max_force_z = 0.5819466433511261
    temp_z = 0.5781
    
    # z_limits = [0.5819352100006313, 0.5854802564152597]
    z_limits = [0.58193, 0.58548]
    # get_z_limits()
    pos, ori = init_robot()
    print(pos)
    # move_z(max_force_z, pos, ori)
    print("Init complete")
    sleep(5)
    alternating_motion_z(z_limits, pos, ori, num_samples=50)

    print_state('Vz')

    '''
    idx_list:
    t   0   x   7   Ax  13  
    Fx  1   y   8   Ay  14
    Fy  2   z   9   Az  15
    Fz  3   Vx  10
    Tx  4   Vy  11
    Ty  5   Vz  12
    Tz  6   
    '''
    







