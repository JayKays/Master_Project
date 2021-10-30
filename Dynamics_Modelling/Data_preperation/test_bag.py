

import yaml
from rosbag.bag import Bag
import matplotlib.pyplot as plt

import numpy as np
import quaternion


TIP_TOPIC = "/panda_simulator/custom_franka_state_controller/tip_state"
VEL_TOPIC = "/panda_simulator/custom_franka_state_controller/robot_state"
'''
Force/pose data: /panda_simulator/custom_franka_state_controller/tip_state
velocities: /panda_simulator/custom_franka_state_controller/robot_state/O_dP_EE
'''

def get_topics(bag_file):
    bag = Bag(bag_file)
    # bag_dict = bag.get_type_and_topic_info()
    # topics = bag.get_type_and_topic_info()[1].keys()
    # types = []
    # for i in range(0,len(topics)):
    #     types.append(bag_dict.values()[i][0])
    
    topics = bag.get_type_and_topic_info()[1].keys()
    types = []
    for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
        types.append(list(bag.get_type_and_topic_info()[1].values())[i][0])

    return topics, types


def force_pose_from_bag(bag_file, state_space_size = 16):

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
            curr_state[0] = t.to_time()
            data[:,idx] = curr_state
            curr_state = np.zeros(state_space_size)

            idx += 1

        itr += 1


    # print("First topic: ", first_topic)
    bag.close()
    return data

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

if __name__ == "__main__":
    test_file   = "2021-10-28-13-40-00.bag"
    test_file   = "2021-10-29-14-53-09.bag"
    test_file   = "2021-10-29-15-09-21.bag"
    test_file   = "2021-10-29-15-33-10.bag"
    out_bag     = "out_test.bag"

    topics, types = get_topics(test_file)

    print(topics)
    print(types)

    rewrite_timestamps(test_file, out_bag)
    data = force_pose_from_bag(out_bag)

    print(data.shape)
    print((data[:,1] - data[:,0]).round(decimals = 8))
    
    np.save("test_data.npy", data)
