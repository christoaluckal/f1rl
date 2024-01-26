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
from geometry_msgs.msg import Point
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


track_file = os.path.join(rp.get_path('f1rl'),rosparam.get_param("track_file"))



csv_f = track_file
x_idx = 0
y_idx = 1
global_path,track_length,x_spline,y_spline = pathgen.get_spline_path(csv_f,x_idx,y_idx)
s_vec = np.linspace(0,track_length,len(global_path))



class VehicleState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vel = 0.0
        self.x_vel = 0
        self.y_vel = 0
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
        # self.odom.append([self.x,self.y,self.x_vel,self.y_vel,self.yaw])
        self.odom.append([self.x,self.y,self.x_vel,self.y_vel])

    def vehicle_state_output(self):
        vehicle_state = np.array([self.x, self.y, self.yaw, self.vel,self.x_vel,self.y_vel])
        return vehicle_state
    
def euclidean_dist(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def distance_to_spline(t,current_x,current_y):
    spline_x, spline_y = x_spline(t), y_spline(t)
    return math.sqrt((spline_x - current_x) ** 2 + (spline_y - current_y) ** 2)
       
def closest_spline_param(current_x,current_y,best_t=0):
    res = minimize(distance_to_spline,x0=best_t, args=(current_x, current_y))
    return res.x

def reset_cb(data):
    global vehicle_state
    if data.data == True:
        vehicle_state = VehicleState()
        print("Resetting Vehicle State")

def step(action):
    ack_msg = AckermannDrive()
    ack_msg.speed = action[0]
    ack_msg.steering_angle = action[1]
    drive_pub.publish(ack_msg)

speeds = np.arange(0,5,0.05)
turns = [-0.3,-0.2,-0.1,0,0.1,0.2,0.3]


if __name__ == "__main__":
    rospy.init_node("car_core",anonymous=True)
    drive_pub = rospy.Publisher(rosparam.get_param("drive_topic"), AckermannDrive, queue_size=1)
    raceline_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)
    spline_marker_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)
    reset_sub = rospy.Subscriber('/reset',Bool,reset_cb)


    rate = rospy.Rate(rosparam.get_param("rate"))
    vehicle_state = VehicleState()
    # N = 5

    done=False

    ref_list = np.array(global_path[:,0:2])

    max_time = 10
    
    while not rospy.is_shutdown():
        
        try:
            reset_cb(Bool(True))
            print("Resetting Vehicle State")
            done=False
            start = time.time()
            reward=0
            poses = []
            while not done:
                if time.time()-start>max_time:
                    break
                else:
                    prev_state = vehicle_state.vehicle_state_output()

                    prev_spline_t = closest_spline_param(prev_state[0],prev_state[1])

                    prev_spline_x, prev_spline_y = x_spline(prev_spline_t), y_spline(prev_spline_t)

                    speed,turn = np.random.choice(speeds),np.random.choice(turns)
                    step([speed,turn])
                    
                    rate.sleep()

                    current_state = vehicle_state.vehicle_state_output()

                    poses.append([current_state[0],current_state[1]])

                    closest_spline_t = closest_spline_param(current_state[0],current_state[1])

                    spline_x, spline_y = x_spline(closest_spline_t), y_spline(closest_spline_t)

                    closest_spline_point = np.array([spline_x,spline_y])

                    if euclidean_dist(current_state[0:2],closest_spline_point)>1:
                        done=True
                        reward-=100

                    if closest_spline_t>track_length-5:
                        done=True
                        reward+=100

                    reward += (closest_spline_t-prev_spline_t) - 1

                    rospy.loginfo("Reward: {}".format(reward))
                

            plt.scatter(ref_list[:,0],ref_list[:,1])
            plt.scatter(np.array(poses)[:,0],np.array(poses)[:,1],c='r')
            plt.show()
                


        except rospy.exceptions.ROSInterruptException:
            
            break
