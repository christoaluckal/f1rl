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

        speeds = np.arange(0,10,0.5)
        turns = np.arange(-0.4,0.41,0.05)

        self.action_map = {}
        counter = 0
        for s in speeds:
            for t in turns:
                self.action_map[counter]=[s,t]
                counter+=1

        self.action_count = len(self.action_map.keys())
        
        

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


    def step(self,idx):
        ack_msg = AckermannDrive()
        action = self.action_map[idx]
        ack_msg.speed = action[0]
        ack_msg.steering_angle = action[1]
        self.drive_pub.publish(ack_msg)
        self.rate.sleep()

    def reset(self):
        # current_state,current_rot = self.vehicle_state.get_state()

        # while(current_state[3]>1e-2):
        #     self.drive_pub.publish(AckermannDrive())
        #     self.rate.sleep()
        #     current_state = self.vehicle_state.get_state()

        # while(math.isclose(current_rot[0],0,abs_tol=0.1) and math.isclose(current_rot[1],0,abs_tol=0.1) and math.isclose(current_rot[2],0,abs_tol=0.1) and math.isclose(current_rot[3],1,abs_tol=0.1)):
        #     pose = Pose()
        #     pose.orientation.w = 1
        #     self.reset_pub.publish(pose)
        #     self.rate.sleep()
        #     current_state,current_rot = self.vehicle_state.get_state()


        # while(current_state[0]>1e-2 and current_state[1]>1e-2):
        #     pose = Pose()
        #     pose.orientation.w = 1
        #     self.reset_pub.publish(pose)
        #     self.rate.sleep()
        #     current_state = self.vehicle_state.get_state()[0]
        counter = 0
        while(counter<10):
            pose = Pose()
            self.reset_pub.publish(pose)
            self.rate.sleep()
            counter+=1

        current_state = self.vehicle_state.get_state()[0]
        while(current_state[3]>1e-2):
            self.drive_pub.publish(AckermannDrive())
            self.rate.sleep()
            current_state = self.vehicle_state.get_state()[0]

def shutdown_hook():
    print("Shutting Down")
    core.drive_pub.publish(AckermannDrive())
    rospy.sleep(1)

if __name__ == "__main__":
    core = CoreCarEnv()
    done=False
    ref_list = np.array(core.global_path[:,0:2])
    max_time = 10

    rospy.on_shutdown(shutdown_hook)
    
    while not rospy.is_shutdown():
        
        try:
            print("Resetting Vehicle State")
            core.reset()
            rospy.sleep(1)
            done=False
            start = time.time()
            reward=0
            poses = []
            while not done:
                if time.time()-start>max_time:
                    break
                else:
                    prev_state = core.vehicle_state.get_state()[0]
                    prev_spline_t = core.closest_spline_param(distance_to_spline,prev_state[0],prev_state[1],core.x_spline,core.y_spline)
                    prev_spline_x, prev_spline_y = core.x_spline(prev_spline_t), core.y_spline(prev_spline_t)

                    action_idx = np.random.randint(0,core.action_count)

                    core.step(action_idx)
                    
                    current_state = core.vehicle_state.get_state()[0]
                    closest_spline_t = core.closest_spline_param(distance_to_spline,current_state[0],current_state[1],core.x_spline,core.y_spline)
                    spline_x, spline_y = core.x_spline(closest_spline_t), core.y_spline(closest_spline_t)


                    closest_spline_point = np.array([spline_x,spline_y])

                    if core.euclidean_dist(current_state[0:2],closest_spline_point)>1:
                        done=True
                        reward-=100

                    if closest_spline_t>core.track_length-5:
                        done=True
                        reward+=100

                    poses.append([current_state[0],current_state[1]])
                    reward += (closest_spline_t-prev_spline_t) - 1

                    # rospy.loginfo("Reward: {}".format(reward))
                

            # plt.scatter(ref_list[:,0],ref_list[:,1])
            # plt.scatter(np.array(poses)[:,0],np.array(poses)[:,1],c='r')
            # plt.show()
                


        except rospy.exceptions.ROSInterruptException:
            
            break
