

# from sys import last_traceback
import yaml
from rosbag.bag import Bag
import matplotlib.pyplot as plt

import numpy as np
import quaternion


TIP_TOPIC = "/panda_simulator/custom_franka_state_controller/tip_state"
VEL_TOPIC = "/panda_simulator/custom_franka_state_controller/robot_state"

'''
Force/pose data : /panda_simulator/custom_franka_state_controller/tip_state
velocity data   : /panda_simulator/custom_franka_state_controller/robot_state/O_dP_EE
'''

def get_topics(bag_file):

    bag = Bag(bag_file)
    topics = bag.get_type_and_topic_info()[1].keys()
    types = []
    for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
        types.append(list(bag.get_type_and_topic_info()[1].values())[i][0])

    return topics, types


def states_from_bag(bag_file, state_space_size = 16):

    bag = Bag(bag_file)
    
    data = np.zeros((state_space_size, bag.get_message_count()//2))

    curr_state = np.zeros(state_space_size) #Temp storage for each full state per timestep

    idx = 0
    itr = 0

    for topic, msg, t in bag.read_messages(topics = [TIP_TOPIC, VEL_TOPIC]):

        if topic == TIP_TOPIC:
           
            cart_pose_trans_mat = np.asarray(msg.O_T_EE).reshape(4,4,order='F')

            cartesian_pose = {
                'position': cart_pose_trans_mat[:3,3],
                'orientation': quaternion.from_rotation_matrix(cart_pose_trans_mat[:3,:3]),
                'orientation_R': cart_pose_trans_mat[:3,:3],
                'transformation_matrix': cart_pose_trans_mat }

            cartesian_effort = {
                'force': np.asarray([ msg.O_F_ext_hat_K.wrench.force.x,
                                    msg.O_F_ext_hat_K.wrench.force.y,
                                    msg.O_F_ext_hat_K.wrench.force.z]),

                'torque': np.asarray([ msg.O_F_ext_hat_K.wrench.torque.x,
                                    msg.O_F_ext_hat_K.wrench.torque.y,
                                    msg.O_F_ext_hat_K.wrench.torque.z])
            }
            # print(topic,"\t", t.to_time()/1000)

            curr_state[1:4]  = cartesian_effort['force']
            curr_state[4:7]  = cartesian_effort['torque']
            curr_state[7:10] = cartesian_pose['position']
            
        if topic == VEL_TOPIC :

            cartesian_velocity = {
                'linear': np.asarray([msg.O_dP_EE[0], msg.O_dP_EE[1], msg.O_dP_EE[2]]),
                'angular': np.asarray([msg.O_dP_EE[3], msg.O_dP_EE[4], msg.O_dP_EE[5]]) }

            # print(np.hstack(([t.to_time()], cartesian_velocity['linear'], cartesian_effort['angular'])))
            curr_state[10:13] = cartesian_velocity['linear']
            curr_state[13:]   = cartesian_velocity['angular']
            

        if itr%2:
            # curr_state[0] = t.to_time()
            data[:,idx] = curr_state
            # curr_state = np.zeros(state_space_size)

            idx += 1
        else:
            curr_state[0] = t.to_time()

        itr += 1
    
    bag.close()
    return data.T

def rewrite_timestamps(input_bag, output_bag):
    '''
    Rewrites input_bag msg timestamps to actual simulation timestamps
    msg headers.
    Rewritten bag is saved in output_bag
    '''
    with Bag(output_bag, 'w') as outbag:
        for topic, msg, t in Bag(input_bag).read_messages():
            # This also replaces tf timestamps under the assumption 
            # that all transforms in the message share the same timestamp
            if topic == "/tf" and msg.transforms:
                outbag.write(topic, msg, msg.transforms[0].header.stamp)
            else:
                outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)

def save_states_from_bag(bag_file, output_file, 
    intermediate_file = "rosbag_record_folder/retimed.bag"):

    '''
    Extracts state data from recorded bag_file, and saves as numpy array to output_file
    Args:
        bag_file: File path to recorded bag file from rosbag
        output_file: File path to save state array
        intermediate_file (optional):  File path to temporarily save bag file with corrected timestamps
    '''

    rewrite_timestamps(bag_file, intermediate_file)
    data = states_from_bag(intermediate_file)

    #Changes timestamps to timesteps
    data[:-1,0] = np.diff(data[:,0])
    data[-1,0] = 0

    if output_file.endswith(".npy"):
        np.save(output_file, data)
    else:
        np.save(output_file + ".npy", data)


if __name__ == "__main__":
    # test_file   = "rosbag_record_folder/_2021-11-02-16-10-34.bag"
    # test_file   = "rosbag_record_folder/_2021-11-02-16-13-34.bag"
    # test_file = "rosbag_record_folder/_2021-11-08-12-39-20.bag"
    # test_file = "rosbag_record_folder/_2021-11-08-12-57-54.bag"
    # test_file = "rosbag_record_folder/_2021-11-08-13-02-47.bag"
    test_file = "rosbag_record_folder/_2021-11-08-13-19-41.bag"
    save_file = "processed_data/stand_still.npy"
    topics, types = get_topics(test_file)
    print(topics)
    print(types)

    save_states_from_bag(test_file, output_file=save_file)

    data = np.load(save_file)
    print(data.shape)
    print(data[:10,0])
    print(data[0,:])
