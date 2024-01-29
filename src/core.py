#!/usr/bin/env python3

'''
Topic: Kinematic Nonlinear Model Predictive Controller for F1tenth simulator
Author: Rongyao Wang
Instiution: Clemson University Mechanical Engineering
'''

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Point,Pose
from visualization_msgs.msg import Marker
import math
from utils import vehicle_coordinate_transformation
import numpy as np
from tf.transformations import euler_from_quaternion
from timeit import default_timer as timer
import os
import pathgen_ds as pathgen
import rosparam
import rospkg
from scipy.optimize import minimize
rp = rospkg.RosPack()
import time
import matplotlib.pyplot as plt
from collections import deque
from networks.policies import EpsilonGreedy
from networks.dqn import NaiveDQN
from networks.nns import General_Network as Network
import torch

experience_buffer = deque(maxlen=int(1e6))
BUFFER_SAMPLE = 256

def distance_to_spline(t,current_x,current_y,x_spline,y_spline):
    spline_x, spline_y = x_spline(t), y_spline(t)
    return math.sqrt((spline_x - current_x) ** 2 + (spline_y - current_y) ** 2)

class CoreCarEnv():
    def __init__(self) -> None:
        rospy.init_node('car_core',anonymous=True)
        self.drive_pub = rospy.Publisher(rosparam.get_param("drive_topic"), AckermannDrive, queue_size=1)
        self.raceline_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)
        self.spline_marker_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)
        self.reset_pub = rospy.Publisher(rosparam.get_param("reset_topic"),Pose,queue_size=1)
        self.vehicle_state = self.VehicleState()
        self.rate = rospy.Rate(rospy.get_param("rate",24))

        track_file = os.path.join(rp.get_path('f1rl'),rosparam.get_param("track_file"))
        csv_f = track_file
        x_idx = 0
        y_idx = 1
        self.global_path,self.track_length,self.x_spline,self.y_spline = pathgen.get_spline_path(csv_f,x_idx,y_idx)

        speeds = np.arange(1,4.1,0.2)
        turns = np.arange(-0.3,0.31,0.01)

        self.action_map = {}
        self.trajectory = []
        self.trajectory = np.array(self.trajectory)
        counter = 0
        for s in speeds:
            for t in turns:
                self.action_map[counter]=[s,t]
                counter+=1

        self.action_count = len(self.action_map.keys())

        self.timesteps = 0
        
        

    class VehicleState:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.yaw = 0.0
            self.vel = 0.0
            self.x_vel = 0
            self.y_vel = 0
            self.qx = 0.0
            self.qy = 0.0
            self.qz = 0.0
            self.qw = 1.0
            self.odom = []
            self.odom_sub = rospy.Subscriber(rosparam.get_param('odom_topic'), Odometry, self.odom_callback)

        def odom_callback(self, msg):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            vel = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            quaternion = (qx,qy,qz,qw)
            euler = euler_from_quaternion(quaternion)
            yaw = euler[2]
            self.x = x
            self.y = y
            self.vel = vel
            self.x_vel = msg.twist.twist.linear.x
            self.y_vel = msg.twist.twist.linear.y
            self.yaw = yaw
            self.odom.append([self.x,self.y,self.x_vel,self.y_vel])

            self.qx = qx
            self.qy = qy
            self.qz = qz
            self.qw = qw

        def get_state(self):
            return [self.x,self.y,self.yaw,self.vel],[self.qx,self.qy,self.qz,self.qw]

    def euclidean_dist(self,p1,p2):
        return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def closest_spline_param(self,fn,current_x,current_y,x_spline,y_spline,best_t=0):
        res = minimize(fn,x0=best_t, args=(current_x, current_y,x_spline,y_spline))
        return res.x


    def reset(self):
        counter = 0
        self.timesteps = 0
        self.trajectory = np.array([])
        while(counter<2):
            pose = Pose()
            self.reset_pub.publish(pose)
            self.rate.sleep()
            counter+=1

        current_state = self.vehicle_state.get_state()[0]
        while(current_state[3]>1e-1):
            self.drive_pub.publish(AckermannDrive())
            self.rate.sleep()
            current_state = self.vehicle_state.get_state()[0]

        current_state = self.vehicle_state.get_state()[0]
        spline_t = self.closest_spline_param(distance_to_spline,current_state[0],current_state[1],self.x_spline,self.y_spline)
        
        state = current_state
        spline_params = np.arange(spline_t,spline_t+2.1,0.4)

        for t in spline_params:
            state.append(current_state[0]-self.x_spline(t))
            state.append(current_state[1]-self.y_spline(t))

        state = np.array(state)
        return state
    
    def get_state_count(self):
        current_state = [0,0,0,0]
        spline_t = self.closest_spline_param(distance_to_spline,current_state[0],current_state[1],self.x_spline,self.y_spline)

        state = current_state
        spline_params = np.arange(spline_t,spline_t+2.1,0.4)

        for t in spline_params:
            # state.append(self.x_spline(t))
            # state.append(self.y_spline(t))
            state.append(current_state[0]-self.x_spline(t))
            state.append(current_state[1]-self.y_spline(t))

        state = np.array(state)
        return state.shape[0]


    def step(self,idx):
        # print(self.trajectory.shape)
        reward = 0
        done = False
        prev_state = core.vehicle_state.get_state()[0]
        prev_spline_t = core.closest_spline_param(distance_to_spline,prev_state[0],prev_state[1],core.x_spline,core.y_spline)


        ack_msg = AckermannDrive()
        action = self.action_map[idx]
        ack_msg.speed = action[0]
        ack_msg.steering_angle = action[1]
        self.drive_pub.publish(ack_msg)
        self.rate.sleep()

        current_state = core.vehicle_state.get_state()[0]
        closest_spline_t = core.closest_spline_param(distance_to_spline,current_state[0],current_state[1],core.x_spline,core.y_spline)
        spline_x, spline_y = core.x_spline(closest_spline_t[0]), core.y_spline(closest_spline_t[0])

        closest_spline_point = np.array([spline_x,spline_y])

        self.trajectory = np.append(self.trajectory,closest_spline_point,axis=0)

        state = current_state
        spline_params = np.arange(closest_spline_t[0],closest_spline_t[0]+2.1,0.4)
        for t in spline_params:
            state.append(current_state[0]-self.x_spline(t))
            state.append(current_state[1]-self.y_spline(t))

        state = np.array(state)

        if core.euclidean_dist(current_state[0:2],closest_spline_point)>3:
            print("Off Track")
            done=True
            reward-=100

        if closest_spline_t>core.track_length-5:
            print("Finished Track")
            done=True
            reward+=100

        current_rot = core.vehicle_state.get_state()[1]

        
        # if math.isclose(current_rot[1],-1,abs_tol=0.1):
        #     print("Flipped")
        #     done=True
        #     reward-=100

        # if self.euclidean_dist(current_state[0:2],prev_state[0:2])<1e-3 and self.timesteps>200:
        #     print("Stuck")
        #     done=True
        #     reward-=100

        self.timesteps+=1
        move_dist = self.euclidean_dist(current_state[0:2],prev_state[0:2])
        # print(f"Move Dist:{move_dist}")
        reward += move_dist*10 - 1e-3*self.timesteps

        return state,reward,done
        



def shutdown_hook():
    print("Shutting Down")
    core.drive_pub.publish(AckermannDrive())
    rospy.sleep(1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = CoreCarEnv()
    done=False
    ref_list = np.array(core.global_path[:,0:2])
    max_time = 25
    epochs = 10000
    max_epsilon = 1
    min_epsilon = 0.01
    decay_rate = (min_epsilon/max_epsilon)**(1/epochs)
    epsilon = max_epsilon
    
    archs = [128,128]

    n_states = core.get_state_count()

    episodes = []
    reward_list = []


    online_network = Network(n_states,core.action_count,archs).to(device)
    target_network = Network(n_states,core.action_count,archs).to(device)

    target_network.load_state_dict(online_network.state_dict())

    discount_rate = 0.99
    rate = 1e-3
    policy = EpsilonGreedy.policy
    trainer = NaiveDQN()



    rospy.on_shutdown(shutdown_hook)
    
    while not rospy.is_shutdown():
        try:
            for i in range(epochs):

                try:
                    # print("Resetting Vehicle State")
                    state = core.reset()
                    rospy.sleep(1)
                    done=False
                    start = time.time()
                    ep_reward=0
                    greeds = 0
                    exploits = 0
                    poses = []
                    while not done:
                        current_state = state
                        if time.time()-start>max_time:
                            done=True
                            reward-=100
                            break
                        else:
                            action_idx,action_type = policy(state,online_network,core.action_count,epsilon,device)

                            if action_type==0:
                                greeds+=1
                            else:
                                exploits+=1

                            next_state,reward,done = core.step(action_idx)

                            experience_buffer.append((current_state,action_idx,reward,next_state,done))

                            online_network = trainer.train(online_network,target_network,experience_buffer,discount_rate,BUFFER_SAMPLE,rate,device)

                            state = next_state
                            ep_reward+=reward

                    epsilon *= decay_rate
                    episodes.append(i)
                    reward_list.append(ep_reward)
                    print(f"Epoch:{i} {greeds}/{exploits} Reward:{ep_reward} Epsilon:{epsilon}")
                
                    with open(os.path.join(rp.get_path('f1rl'),'src/dqn_reward.txt'),'a') as f:
                        f.write(f"{i}\t{ep_reward}\n")
                        


                except (KeyboardInterrupt,rospy.ROSInterruptException):
                    raise Exception("Shutting Down")
                
        except Exception as e:
            break



    