import gym
from franka_interface import ArmInterface
import numpy as np
import rospy
from gym import logger, spaces
from scipy.stats import truncnorm
import time
from gym.utils import seeding
import sys
sys.path.append("/home/jaykay/Master_Project/src/Compliant_control/gym-panda/VIC")
from gym_panda.envs import VIC_func as func
from gym_panda.envs import VIC_config as cfg
# from gym_panda.envs.VIC_ObsSpace import ObservationSpace
import matplotlib.pyplot as plt
import time
import random
import quaternion
import matplotlib.pyplot as plt

from time import sleep
np.random.seed(0)


""" GENERAL COMMENTS 

1) Gazebo must be setup before the training starts (object in place + servers running)


"""


class VIC_Env(gym.Env):

    def __init__(self):
        # only in __init()
        self.name = "VIC"
        self.learning = False
        self.sim = cfg.SIM_STATUS
        # self.action_space = spaces.Box(low= np.array([cfg.GAMMA_B_LOWER,cfg.GAMMA_K_LOWER,cfg.KP_POS_LOWER]), high = np.array([cfg.GAMMA_B_UPPER,cfg.GAMMA_K_UPPER,cfg.KP_POS_UPPER]))
        # two dim action space
        """
        self.action_space = spaces.Box(low= np.array([cfg.GAMMA_B_LOWER,cfg.GAMMA_K_LOWER]), high = np.array([cfg.GAMMA_B_UPPER,cfg.GAMMA_K_UPPER])) 
        self.observation_space_container= ObservationSpace() 
        self.observation_space = self.observation_space_container.get_space_box()
        """
        self.action_space = spaces.Box(low= np.array([cfg.GAMMA_B_LOWER,cfg.GAMMA_K_LOWER]), \
                                       high = np.array([cfg.GAMMA_B_UPPER,cfg.GAMMA_K_UPPER]))
        #print("space: ", self.action_space.low[0])
        '''
        self.observation_space = spaces.Box(low= np.array([cfg.LOWER_Fz, cfg.LOWER_Z_ERROR, cfg.LOWER_Vz, cfg.LOWER_X_ERROR,\
                                        cfg.LOWER_Y_ERROR, cfg.LOWER_Fx, cfg.LOWER_Fy, cfg.LOWER_Tx, cfg.LOWER_Ty, \
                                        cfg.LOWER_Tz, cfg.LOWER_Vx, cfg.LOWER_Vy, cfg.LOWER_Ax, cfg.LOWER_Ay, cfg.LOWER_Az]), \
                                        high = np.array([cfg.UPPER_Fz, cfg.UPPER_Z_ERROR, cfg.UPPER_Vz , cfg.UPPER_X_ERROR, \
                                        cfg.UPPER_Y_ERROR, cfg.UPPER_Fx, cfg.UPPER_Fy, cfg.UPPER_Tx, cfg.UPPER_Ty, cfg.UPPER_Tz,\
                                        cfg.UPPER_Vx, cfg.UPPER_Vy, cfg.UPPER_Ax, cfg.UPPER_Ay, cfg.UPPER_Az])) '''
        self.observation_space = spaces.Box(low= np.array([cfg.LOWER_Fz, cfg.LOWER_Z_ERROR, cfg.LOWER_Vz, cfg.LOWER_X_ERROR,\
                                        cfg.LOWER_Y_ERROR]), \
                                        high = np.array([cfg.UPPER_Fz, cfg.UPPER_Z_ERROR, cfg.UPPER_Vz , cfg.UPPER_X_ERROR, \
                                        cfg.UPPER_Y_ERROR]))

        self.max_num_it = cfg.MAX_NUM_IT
        #print(self.max_num_it)
        print("max ",self.max_num_it)

        # setup
        rospy.init_node("VIC")
        self.rate = rospy.Rate(cfg.PUBLISH_RATE)
        self.robot = ArmInterface()
        self.joint_names = self.robot.joint_names()

        self.M = cfg.M
        self.B = cfg.B
        self.K = cfg.K

        # also in reset()
        """
        random_number = random.uniform(-1,1)
        if random_number <= 0:
            start_neutral = True
        else:
            start_neutral = False

        self.robot.move_to_start(cfg.ALTERNATIVE_START, cfg.RED_START ,self.sim, start_neutral=start_neutral)
        """
        # Moving to correct starting position in reset() instead
        self.move_to_start(cfg.cartboard, self.sim)
        #print(self.robot.endpoint_effort()['force'][2])
        self.gamma = np.identity(18)
        self.gamma[8, 8] = cfg.GAMMA_B_INIT
        self.gamma[14, 14] = cfg.GAMMA_K_INIT
        self.init_action = np.array([cfg.GAMMA_B_INIT, cfg.GAMMA_B_INIT])
        # self.Kp_pos = cfg.KP_POS

        self.lam = np.zeros(18)
        # set desired pose/force trajectory
        # self.f_d = func.generate_Fd_steep(self.max_num_it, cfg.Fd,cfg.T)
        self.f_d = func.generate_Fd_wave(self.max_num_it, cfg.Fd)
        # self.f_d[2, :] = cfg.Fd
        plt.plot(np.arange(self.f_d.shape[1])/100, self.f_d[2,:])
        plt.title("Desired Force",fontsize = 14)
        plt.xlabel("Time [s]",fontsize = 12)
        plt.ylabel("Force [N]",fontsize = 12)
        plt.savefig("figures_for_report/F_d.eps", format = "eps")
        plt.show()
        # print(self.f_d)
        self.Rot_d = self.robot.endpoint_pose()['orientation_R']
        self.goal_ori = np.asarray(self.robot.endpoint_pose()['orientation']) # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        #self.goal_ori = self.robot.endpoint_pose()['orientation']
        self.x_d_ddot, self.x_d_dot, self.p_d = func.generate_desired_trajectory_tc(self.robot, self.max_num_it, cfg.T,move_in_x=True)
        plt.plot(np.arange(self.p_d.shape[1])/100, self.p_d[0,:])
        plt.title("Desired x-position", fontsize = 14)
        plt.xlabel("Time [s]",fontsize = 12)
        plt.ylabel("x-position",fontsize = 12)
        plt.savefig("figures_for_report/x_d.eps", format = "eps")
        plt.show()
        sleep(5)
        print(self.x_d_ddot)
        self.iteration = 0
        self.time_per_iteration = np.zeros(self.max_num_it)

        self.x_history = np.zeros((6, self.max_num_it))
        self.x_dot_history = np.zeros((6, self.max_num_it))
        self.p_hist = np.zeros((3, self.max_num_it))
        self.Fz_history = np.zeros(self.max_num_it)
        self.h_e_hist = np.zeros((6, self.max_num_it))
        self.gamma_B_hist = np.zeros(self.max_num_it)
        self.gamma_K_hist = np.zeros(self.max_num_it)
        self.Kp_pos_hist = np.zeros(self.max_num_it)
        self.Kp_z_hist = np.zeros(self.max_num_it)
        self.Kd_z_hist = np.zeros(self.max_num_it)

        #self.reset()
        self.Rot_e, self.p, self.x, self.x_dot, self.x_history, self.x_dot_history, self.delta_x, self.jacobian, self.robot_inertia,\
        self.Fz, self.F_ext, self.F_ext_2D, self.coriolis_comp = self.robot.get_VIC_states(self.iteration, self.time_per_iteration, \
                                                self.p_d[:, self.iteration], self.goal_ori, self.x_history, self.x_dot_history, self.sim)
        self.Fz_offset = 0#self.Fz
        self.p_z_init = self.p[2]

        # array with data meant for plotting
        self.data_for_plotting = np.zeros((17, self.max_num_it))

        self.h_c = None
        self.torques = None

    def move_to_start(self, alternative_position, sim):
        if sim:
            self.robot.move_to_neutral()
        else:
            self.robot.move_to_joint_positions(alternative_position)

    def set_learning(self, ):
        self.learning = True

    def set_not_learning(self):
        self.learning = False

    def quatdiff_in_euler_radians(self, quat_curr, quat_des):
        curr_mat = quaternion.as_rotation_matrix(quat_curr)
        des_mat = quaternion.as_rotation_matrix(quat_des)
        rel_mat = des_mat.T.dot(curr_mat)
        rel_quat = quaternion.from_rotation_matrix(rel_mat)
        vec = quaternion.as_float_array(rel_quat)[1:]
        if rel_quat.w < 0.0:
            vec = -vec
        return -des_mat.dot(vec)

    def get_x(self, goal_ori):
        pos_x = self.robot.endpoint_pose()['position']
        rel_ori = self.quatdiff_in_euler_radians(np.asarray(self.robot.endpoint_pose()['orientation']),goal_ori)  # used to be opposite
        return np.append(pos_x, rel_ori)

    def get_derivative_of_vector(self, history, iteration, time_per_iteration):
        size = history.shape[0]
        if iteration > 0:
            T = float(time_per_iteration[int(iteration)] - time_per_iteration[int(iteration - 1)])
            if T > 0:
                return np.subtract(history[:, iteration], history[:, iteration - 1]) / T
        return np.zeros(size)

    def get_delta_x(self,x, p_d, two_dim=False):
        delta_pos = p_d - x[:3]
        delta_ori = x[3:]
        if two_dim == True:
            return np.array([np.append(delta_pos, delta_ori)]).reshape([6, 1])
        else:
            return np.append(delta_pos, delta_ori)

    def fetch_states(self, i, time_per_iteration, p_d, sim):
        x = self.get_x(self.goal_ori)
        self.x_history[:, i] = x
        p = x[:3]
        jacobian = self.robot.zero_jacobian()
        robot_inertia = self.robot.joint_inertia_matrix()
        Fz = self.robot.endpoint_effort()['force'][2]
        F_ext = np.array([0,0,Fz,0,0,0])
        F_ext_2D = F_ext.reshape([6, 1])
        if sim:
            x_dot = self.get_derivative_of_vector(self.x_history, i, time_per_iteration)
        else:
            x_dot = np.append(self.robot.endpoint_velocity()['linear'], self.robot.endpoint_velocity()['angular'])
        self.x_dot_history[:, i] = x_dot
        delta_x = self.get_delta_x(x, p_d)
        Rot_e = self.robot.endpoint_pose()['orientation_R']
        return Rot_e, p, x, x_dot, delta_x, jacobian, robot_inertia, Fz, F_ext, F_ext_2D

    def get_x_dot_delta(self,x_d_dot, x_dot, two_dim=True):
        if two_dim == True:
            return (x_d_dot - x_dot).reshape([6, 1])
        else:
            return x_d_dot - x_dot

    def get_x_ddot_delta(self,x_d_ddot, v_history, i, time_per_iteration):
        a = self.get_derivative_of_vector(v_history, i, time_per_iteration)
        return x_d_ddot - a

    def get_xi(self, x_dot, x_d_dot, x_d_ddot, delta_x, x_dot_history, i, time_per_iteration):
        E = -delta_x
        E_dot = -self.get_x_dot_delta(x_d_dot, x_dot, two_dim=False)
        E_ddot = -self.get_x_ddot_delta(x_d_ddot, x_dot_history, i, time_per_iteration)
        E_diag = np.diagflat(E)
        E_dot_diag = np.diagflat(E_dot)
        E_ddot_diag = np.diagflat(E_ddot)
        return np.block([E_ddot_diag, E_dot_diag, E_diag])

    def get_lambda_dot(self, gamma, xi, K_v, P, F_d, F_ext_2D, i, time_per_iteration, T):
        if i > 0:
            T = float(time_per_iteration[i] - time_per_iteration[i - 1])
        return np.linalg.multi_dot(
            [-np.linalg.inv(gamma), xi.T, np.linalg.inv(K_v), P, F_ext_2D - F_d.reshape([6, 1])]) * T

    def update_MBK_hat(self, lam, M, B, K):
        M_hat = M  # + np.diagflat(lam[0:6]) M is chosen to be constant
        B_hat = B + np.diagflat(lam[6:12])
        K_hat = K + np.diagflat(lam[12:18])
        # ensure_limits(1,5000,M_hat)
        B_hat = np.clip(B_hat, cfg.B_hat_lower, cfg.B_hat_upper)
        K_hat = np.clip(K_hat, cfg.K_hat_lower, cfg.K_hat_upper)
        return M_hat, B_hat, K_hat

    def skew(self,vector):
        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])

    def get_K_Pt_dot(self, R_d, K_pt, R_e):
        return np.array([0.5 * np.linalg.multi_dot([R_d, K_pt, R_d.T]) + 0.5 * np.linalg.multi_dot([R_e, K_pt, R_e.T])])

    def get_K_Pt_ddot(self,p_d, R_d, K_pt, delta_x):
        return np.array([0.5 * np.linalg.multi_dot([self.skew(delta_x[:3]), R_d, K_pt, R_d.T])])

    def E_quat(self, quat_n, quat_e):
        return np.dot(quat_n, np.identity(3)) - self.skew(quat_e)

    def get_K_Po_dot(self, quat_n, quat_e, R_e, K_po):
        return np.array([2 * np.linalg.multi_dot([self.E_quat(quat_n, quat_e).T, R_e, K_po, R_e.T])])

    def get_h_delta(self, K_pt_dot, K_pt_ddot, p_delta, K_po_dot, quat_e):
        f_delta_t = np.array([np.dot(K_pt_dot, p_delta)])
        m_delta_t = np.array([np.dot(K_pt_ddot, p_delta)])
        null = np.zeros((3, 1))
        m_delta_o = np.array([np.dot(K_po_dot, quat_e)])

        return np.array([np.append(f_delta_t.T, m_delta_t.T)]).T + np.array([np.append(null.T, m_delta_o.T)]).T

    # Return the cartesian (task-space) inertia of the manipulator [alternatively the inverse of it]
    def get_W(self, jacobian, robot_inertia, inv=False):
        W = np.linalg.multi_dot([jacobian, np.linalg.inv(robot_inertia), jacobian.T])
        if inv == True:
            return np.linalg.inv(W)
        else:
            return W

    def perform_torque_DeSchutter(self, M, B, K, x_d_ddot, x_d_dot, x_dot, delta_x, p_d, Rot_e, Rot_d, F_ext_2D, jacobian,
                                  robot_inertia, joint_names, sim):  # must include Rot_d
        Rot_e_bigdim = np.block([[Rot_e,np.zeros((3,3))],[np.zeros((3,3)),Rot_e]])
        Rot_e_dot = np.dot(self.skew(x_dot[3:]), Rot_e)  # not a 100 % sure about this one
        Rot_e_dot_bigdim = np.block([[Rot_e_dot,np.zeros((3,3))],[np.zeros((3,3)),Rot_e_dot]])

        quat = quaternion.from_rotation_matrix(np.dot(Rot_e.T, Rot_d))  # orientational displacement represented as a unit quaternion
        quat_e_e = np.array([quat.x, quat.y, quat.z])  # vector part of the unit quaternion in the frame of the end effector
        quat_e = np.dot(Rot_e.T, quat_e_e)  # ... in the base frame
        quat_n = quat.w
        p_delta = delta_x[:3]
        K_Pt_dot = self.get_K_Pt_dot(Rot_d, K[:3, :3], Rot_e)
        K_Pt_ddot = self.get_K_Pt_ddot(p_d, Rot_d, K[:3, :3], delta_x)
        K_Po_dot = self.get_K_Po_dot(quat_n, quat_e, Rot_e, K[3:, 3:])

        h_delta_e = np.array(np.dot(Rot_e_bigdim, self.get_h_delta(K_Pt_dot, K_Pt_ddot, p_delta, K_Po_dot, quat_e))).\
                                                                                                        reshape([6, 1])
        h_e_e = np.array(np.dot(Rot_e_bigdim, F_ext_2D))
        a_d_e = np.dot(Rot_e_bigdim, x_d_ddot).reshape([6, 1])
        v_d_e = np.dot(Rot_e_bigdim, x_d_dot).reshape([6, 1])
        alpha_e = a_d_e + np.dot(np.linalg.inv(M), (np.dot(B,v_d_e.reshape([6, 1]) - np.dot(Rot_e_bigdim, x_dot).\
                                                           reshape([6, 1])) + h_delta_e - h_e_e)).reshape([6, 1])
        alpha = np.dot(Rot_e_bigdim.T, alpha_e).reshape([6, 1]) + np.dot(Rot_e_dot_bigdim.T,
                                                                         np.dot(Rot_e_bigdim, x_dot)).reshape([6, 1])
        torque = np.linalg.multi_dot([jacobian.T, self.get_W(jacobian, robot_inertia, inv=True), alpha]).\
                                    reshape((7, 1)) + np.array(self.robot.coriolis_comp().reshape((7, 1))) \
                                                                + np.dot(jacobian.T, F_ext_2D).reshape((7, 1))
        #tau = J.T * h_c
        # print(jacobian.shape)
        # print(torque.shape)

        self.h_c, _, _, _ = np.linalg.lstsq(jacobian.T, torque)
        self.torques = torque
        # print(self.h_c.shape)

        self.robot.set_joint_torques(dict(list(zip(joint_names, torque))))

    def step(self, action):

        #self.learning = False
        # perform action
        self.gamma[8, 8] = action[0]  # gamma B
        self.gamma[14, 14] = action[1]  # gamma K
        '''
        if self.learning:
            #print("learning")
            self.gamma[8, 8] = action[0]# gamma B
            self.gamma[14, 14] = action[1]  # gamma K
        else:
            #print("sdefsf")
            self.gamma[8, 8] = 0.001 #truncnorm.rvs(self.action_space.low[0], self.action_space.high[0]) # gamma B
            self.gamma[14, 14] = 0.00001#truncnorm.rvs(self.action_space.low[1], self.action_space.high[1])
        '''
        # updating states
        self.time_per_iteration[self.iteration] = rospy.get_time()
        Rot_e, p, x, x_dot, delta_x, jacobian, robot_inertia, Fz, F_ext, F_ext_2D = \
                self.fetch_states(self.iteration, self.time_per_iteration, self.p_d[:, self.iteration], self.sim)
        xi = self.get_xi(x_dot, self.x_d_dot[:, self.iteration], self.x_d_ddot[:, self.iteration], delta_x, \
                                                     self.x_dot_history, self.iteration, self.time_per_iteration)
        #print(self.get_lambda_dot(self.gamma,xi,cfg.K_v,cfg.P,self.f_d[:,self.iteration],\
                                            #F_ext_2D, self.iteration,self.time_per_iteration,cfg.T).reshape([18,1])[14])

        self.lam = self.lam.reshape([18,1]) + self.get_lambda_dot(self.gamma,xi,cfg.K_v,cfg.P,self.f_d[:,self.iteration],\
                                            F_ext_2D, self.iteration,self.time_per_iteration,cfg.T).reshape([18,1])

        M_hat, B_hat, K_hat = self.update_MBK_hat(self.lam, self.M, self.B, self.K)
        self.perform_torque_DeSchutter(M_hat, B_hat, K_hat, self.x_d_ddot[:,self.iteration], self.x_d_dot[:,self.iteration]\
                                        ,x_dot, delta_x, self.p_d[:,self.iteration], Rot_e, self.Rot_d,F_ext_2D, \
                                        jacobian, robot_inertia,self.joint_names, self.sim)
        start = time.time()
        time.sleep(1/100)
        self.state_dict = self.robot.get_full_state_space()
        #rate = rospy.Rate(100)
        #self.rate.sleep()
        print("time taken is: ", 1 / (time.time() - start))
        # add new state to history
        self.Fz_history[self.iteration] = self.Fz
        self.h_e_hist[:, self.iteration] = self.F_ext
        self.p_hist[:, self.iteration] = self.p



        # add action to record (plotting purposes)
        self.gamma_B_hist[self.iteration] = action[0]
        self.gamma_K_hist[self.iteration] = action[1]
        self.Kp_pos_hist[self.iteration] = cfg.Kp  # Not a part of the learned strategy anymore
        self.Kp_z_hist[self.iteration] = K_hat[2, 2]
        self.Kd_z_hist[self.iteration] = B_hat[2, 2]


        #rate = self.rate

        #self.robot.set_joint_velocities({'panda_joint1': 0, 'panda_joint2': 0, 'panda_joint3': 0, 'panda_joint4': 0, \
                     #'panda_joint5': 0, 'panda_joint6': 0, 'panda_joint7': 0})
        '''
        self.state = self.robot.get_full_state_space(self.p_z_init, self.Fz_offset, self.f_d[2, self.iteration],
                                                      self.p_d[:, self.iteration], self.h_e_hist, self.iteration,
                                                      self.time_per_iteration) '''

        '''
        self.state = np.array([self.state_dict["Fz"]-self.Fz_offset, self.state_dict["z"]-self.p_z_init, self.state_dict["Vz"], \
                        self.state_dict["x"]-self.p_d[0, self.iteration], self.state_dict["y"]-self.p_d[1, self.iteration], \
                        self.state_dict["Fx"], self.state_dict["Fy"], self.state_dict["Tx"], self.state_dict["Ty"], \
                        self.state_dict["Tz"], self.state_dict["Vx"], self.state_dict["Vy"], self.state_dict["Ax"], \
                        self.state_dict["Ay"], self.state_dict["Az"]])'''
        self.state = np.array([self.state_dict["Fz"]-self.Fz_offset, self.state_dict["z"]-self.p_z_init, self.state_dict["Vz"], \
                        self.state_dict["x"]-self.p_d[0, self.iteration], self.state_dict["y"]-self.p_d[1, self.iteration]])
        # Gym-related..
        reward = 0
        # print(self.iteration, action,[self.K[2, 2], self.B[2, 2]], [K_hat[2, 2], B_hat[2, 2]], self.p_z_init, self.f_d[2, self.iteration],
        #       self.state[0])
        print(f"Gammas: B: {self.gamma_B_hist[self.iteration]}, K: {self.gamma_K_hist[self.iteration]}")
        if (self.iteration >= self.max_num_it) or (np.abs(self.state[1]) > 0.05) or (np.abs(self.state[3]) > 0.02) or \
            (np.abs(self.state[4]) > 0.05) or (np.abs(self.f_d[2, self.iteration]-self.state[0]) > 10) or (self.state[0] < 0.5):
            done = True#(self.iteration >= self.max_num_it)
            print("Fz:", self.state[0], " ez:", np.abs(self.p_z_init - self.p[2]),np.abs(self.state[1]), " ex:", np.abs(self.state[3]), " ey:",
                  np.abs(self.state[4]))
            self.update_data_for_plotting()
            placeholder = self.data_for_plotting
        if (self.iteration >= self.max_num_it-1):
            done = True
        else:
            done = False
            placeholder = None
        if not done:
            F_reward = np.exp(-np.square(3*(self.f_d[2, self.iteration]-self.state[0])))
            x_reward = np.exp(-np.square(300*self.state[3]))
            y_reward = np.exp(-np.square(300*self.state[4]))
            #print(self.state[0],F_reward, self.state[3], x_reward, self.state[4], y_reward)
            reward = 0.5*F_reward + 0.3*x_reward + 0.2*y_reward
        else:
            reward = 0
        self.iteration += 1
        return self.state, reward, done, {}

    def reset(self):
        print("resetting")
        # time.sleep(30)
        """
        self.gamma = np.identity(18)
        self.gamma[8,8] = cfg.GAMMA_B_INIT
        self.gamma[14,14] = cfg.GAMMA_K_INIT
        #self.Kp_pos = cfg.KP_POS_INIT
        """
        '''
        index = np.random.randint(0, (0.9 * self.max_num_it))
        self.iteration = index
        self.lam = np.zeros(18)
        #self.robot.move_to_start(cfg.ALTERNATIVE_START, cfg.RED_START, self.sim)
        self.initialize_robot_pose()'''
        # set desired pose/force trajectory
        index = 0  # np.random.randint(0, (0.9 * self.max_num_it))
        self.iteration = index
        self.lam = np.zeros(18)
        self.move_to_start(cfg.cartboard, self.sim)
        # self.initialize_robot_pose()
        # set desired pose/force trajectory
        #f_d = np.concatenate([self.robot.endpoint_effort()['force'], self.robot.endpoint_effort()['torque']])
        #f_d[2] = 5
        #self.f_d = np.transpose(np.tile(f_d, (self.max_num_it, 1)))
        #self.Rot_d = self.robot.endpoint_pose()['orientation_R']
        #self.f_d = func.generate_Fd_constant(self.max_num_it, cfg.Fd)  # func.generate_Fd_steep(self.max_num_it,cfg.T,cfg.Fd)
        #self.goal_ori = self.robot.endpoint_pose()['orientation']  # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
        #self.x_d_ddot, self.x_d_dot, self.p_d = func.generate_desired_trajectory(self.robot, self.max_num_it, cfg.T, self.sim, move_in_x=True)
        self.time_per_iteration = np.zeros(self.max_num_it)
        self.x_history = np.zeros((6, self.max_num_it))
        self.x_dot_history = np.zeros((6, self.max_num_it))
        self.p_hist = np.zeros((3, self.max_num_it))
        self.Fz_history = np.zeros(self.max_num_it)
        self.h_e_hist = np.zeros((6, self.max_num_it))
        self.gamma_B_hist = np.zeros(self.max_num_it)
        self.gamma_K_hist = np.zeros(self.max_num_it)
        self.Kp_pos_hist = np.zeros(self.max_num_it)
        self.Kp_z_hist = np.zeros(self.max_num_it)
        self.Kd_z_hist = np.zeros(self.max_num_it)
        '''
        self.Rot_e, self.p, self.x, self.x_dot, self.x_history, self.x_dot_history, self.delta_x, self.jacobian, self.robot_inertia, \
        self.Fz, self.F_ext, self.F_ext_2D, self.coriolis_comp = self.robot.get_VIC_states( self.iteration, self.time_per_iteration, \
                                                    self.p_d[:, self.iteration], self.goal_ori, self.x_history, self.x_dot_history, self.sim)
        self.Fz_offset = 0# self.Fz
        self.p_z_init = self.p[2]
        '''
        # array with data meant for plotting
        self.data_for_plotting = np.zeros((17, self.max_num_it))    
        self.state_dict = self.robot.get_full_state_space()
        '''
        self.state = np.array([self.state_dict["Fz"]-self.Fz_offset, self.state_dict["z"]-self.p_z_init, self.state_dict["Vz"], \
                        self.state_dict["x"]-self.p_d[0, self.iteration], self.state_dict["y"]-self.p_d[1, self.iteration], \
                        self.state_dict["Fx"], self.state_dict["Fy"], self.state_dict["Tx"], self.state_dict["Ty"], \
                        self.state_dict["Tz"], self.state_dict["Vx"], self.state_dict["Vy"], self.state_dict["Ax"], \
                        self.state_dict["Ay"], self.state_dict["Az"]])'''
        self.state = np.array([self.state_dict["Fz"]-self.Fz_offset, self.state_dict["z"]-self.p_z_init, self.state_dict["Vz"], \
                        self.state_dict["x"]-self.p_d[0, self.iteration], self.state_dict["y"]-self.p_d[1, self.iteration]])
        self.steps_beyond_done = None
        return self.state

    def initialize_robot_pose(self):
        #self.robot.move_to_start(cfg.ALTERNATIVE_START, cfg.RED_START, self.sim)
        self.robot.move_to_joint_positions(cfg.ALTERNATIVE_START)
        #print(index)
        pose = self.p_d[:, self.iteration]
        self.robot.move_to_cartesian_pose(pose)
        j_ang = (self.robot.joint_angles())
        j_ang['panda_joint7'] = cfg.ALTERNATIVE_START['panda_joint7']
        # robot.move_to_start(cfg.ALTERNATIVE_START, cfg.RED_START ,sim)
        #print("before: ", self.robot.endpoint_effort())
        self.robot.move_to_joint_positions(j_ang)
        #self.robot.move_to_cartesian_pose(pose)
        print(self.robot.endpoint_effort())

    def update_data_for_plotting(self):
        self.data_for_plotting[0, :] = self.Fz_history  # force in z
        self.data_for_plotting[1, :] = self.f_d[2, :]  # desired force in z
        self.data_for_plotting[2, :] = self.p_hist[0, :]  # x pos
        self.data_for_plotting[3, :] = self.p_hist[1, :]  # y pos
        self.data_for_plotting[4, :] = self.p_hist[2, :]  # z pos
        self.data_for_plotting[5, :] = self.p_d[0]  # desired x position
        self.data_for_plotting[6, :] = self.p_d[1]  # desired y position
        self.data_for_plotting[7, :] = self.p_d[2]  # desired z position (below the surface)
        self.data_for_plotting[8, :] = self.x_history[3, :]  # error orientation (x)
        self.data_for_plotting[9, :] = self.x_history[4, :]  # error orientation (y)
        self.data_for_plotting[10, :] = self.x_history[5, :]  # error orientation (z)
        self.data_for_plotting[11, :] = self.time_per_iteration  # how long time did each iteration take
        self.data_for_plotting[12, :] = self.gamma_B_hist  # adaptive rate of damping in z
        self.data_for_plotting[13, :] = self.gamma_K_hist  # adaptive rate of stiffness in z
        self.data_for_plotting[14, :] = self.Kd_z_hist  # damping in z over time
        self.data_for_plotting[15, :] = self.Kp_z_hist  # stiffness in z over time
        self.data_for_plotting[16, :] = self.Kp_pos_hist  # stiffness in x and y over time

    def set_fd_constant(self, force):
        
        self.f_d = force

    def get_wrench(self):
        return self.h_c
    
    def get_torque(self):
        return self.torques

class Normalised_VIC_Env():
    def __init__(self, env_id, m, std):
        self.env = gym.make(env_id)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x - self.m, self.std)

    def step(self, action):
        ob, r, done, plot_data = self.env.step(action)
        return self.state_trans(ob), r, done, plot_data

    def reset(self):
        ob = self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()

def state_list_from_dict(state_dict):
    state_list = np.array([state_dict["Fx"], state_dict["Fy"], state_dict["Fz"] , state_dict["Tx"], \
                state_dict["Ty"], state_dict["Tz"], state_dict["x"], \
                state_dict["y"] , state_dict["z"] , state_dict["Vx"], \
                state_dict["Vy"], state_dict["Vz"], state_dict["Ax"], state_dict["Ay"],
                state_dict["Az"]])
    return state_list

if __name__ == "__main__":

    sim = True
    rospy.init_node("VIC")
    rate = rospy.Rate(cfg.PUBLISH_RATE)
    robot = ArmInterface()
    joint_names = robot.joint_names()
    #print(joint_names)
    env = VIC_Env()
    publish_rate = 100
    max_num_it = cfg.duration * publish_rate
    env.set_not_learning()

    num_trials = 1
    forces = np.zeros(num_trials)
    for n in range(num_trials):
        env.reset()
        state_data = np.zeros((max_num_it,16))
        # fd = func.generate_Fd_random(env.max_num_it, cfg.Fd, cfg.T, slope_prob = 0, Fmin = 3, Fmax = 10)
        # env.set_fd_constant(cfg.Fd)
        # forces[n] = fd[2,0]
        r = rospy.Rate(publish_rate)
        for i in range(max_num_it):
            start = time.time()
            action = np.array([0.0001*10 , 0.0001/10 ])
            # action = np.random.uniform(cfg.GAMMA_B_LOWER, cfg.GAMMA_B_UPPER, 2)
            env.step(action)
            timestep = time.time() - start

            # state_data[i,0] = timestep
            # state_data[i,1:] = state_list_from_dict(robot.get_full_state_space())
            # print("time taken is: ", 1 / (timestep))
            # r.sleep()
        print(f"Sample {n+1} of {num_trials} complete!")
    # np.save("test_VIC_rec2.npy", state_data)
    print("Forces for each trajectory: ",forces)
    print(f"Finished {num_trials} trajectories")
