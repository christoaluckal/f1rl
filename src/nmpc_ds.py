#!/usr/bin/env python3

'''
Topic: Kinematic Nonlinear Model Predictive Controller for F1tenth simulator
Author: Rongyao Wang
Instiution: Clemson University Mechanical Engineering
'''

import rospy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import math
from utils import vehicle_coordinate_transformation
import numpy as np
from tf.transformations import euler_from_quaternion
import casadi
from timeit import default_timer as timer
import csv
import os
import pathgen_ds as pathgen
import sys
import matplotlib.pyplot as plt
from std_srvs.srv import SetBool
import rosparam
import rospkg
from scipy.optimize import minimize_scalar,minimize
from scipy import signal
from scipy.interpolate import CubicSpline
rp = rospkg.RosPack()

global_speed = rosparam.get_param("global_speed")
N = rosparam.get_param("N")
future_time = rosparam.get_param("future_time")
WB = rosparam.get_param("WB")
vel_max = rosparam.get_param("vel_max")
vel_min = rosparam.get_param("vel_min")
max_steer = rosparam.get_param("max_steer")
max_acc = rosparam.get_param("max_acc")

track_file = os.path.join(rp.get_path('nonlinear_mpc'),rosparam.get_param("track_file"))

qx = rosparam.get_param("Q")[0]
qy = rosparam.get_param("Q")[1]
q_yaw = rosparam.get_param("Q")[2]
q_vel = rosparam.get_param("Q")[3]
r_acc = rosparam.get_param("R")[0]
r_steer = rosparam.get_param("R")[1]
u_acc = rosparam.get_param("U")[0]
u_steer = rosparam.get_param("U")[1]
is_raceline = rosparam.get_param("is_raceline")

N_x = 4
N_u = 2

dt = future_time/N



def sawtooth_wave(a,b,t,count):
    t = np.linspace(0, t,count)
    chunk_size = 30

    sawtooth_wave = signal.sawtooth(2 * np.pi * (1/chunk_size) * t, 0.5)
    sawtooth_wave = (b - a) * (sawtooth_wave + 1) / 2 + a
    return sawtooth_wave


if not is_raceline:
    csv_f = track_file
    global_path,track_length,x_spline,y_spline,_ = pathgen.get_spline_path(csv_f)
    v_profile = sawtooth_wave(2,10,track_length,len(global_path))
    v_profile = np.array(v_profile)
    v_spline = CubicSpline(np.linspace(0,track_length,len(global_path)),v_profile)
    s_vec = np.linspace(0,track_length,len(global_path))
    plt.plot(s_vec,v_profile)
    plt.show()
else:
    csv_f = track_file
    x_idx = 6
    y_idx = 7
    v_idx = 8
    global_path,track_length,x_spline,y_spline,v_spline = pathgen.get_spline_path(csv_f,x_idx,y_idx,v_idx)
    s_vec = np.linspace(0,track_length,len(global_path))
    plt.plot(s_vec,v_spline(s_vec))
    plt.show()
    


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


def nonlinear_kinematic_mpc_solver(current_state,last_t=0):
    opti = casadi.Opti()
    
    x = opti.variable(N_x, N+1)
    u = opti.variable(N_u, N)

    x_0 = [0,0,0,current_state[3]]
    x_0 = casadi.MX(x_0)

    Q = casadi.diag([qx, qy, q_yaw, q_vel])
    R = casadi.diag([r_acc, r_steer])
    cost = 0

    current_speed = current_state[3]

    ref_t_list = []

    init_t = None

    for t in range(N-1):
        cost += u[:, t].T @ R @ u[:, t]
        if t != 0:
            if current_speed is not None:
                init_t = closest_spline_param(current_state[0],current_state[1],last_t)
                desired_speed = v_spline(init_t)
                ds = current_speed*dt+(1/2)*(desired_speed-current_speed)*dt
                next_t = (init_t+ds)%track_length
                next_x = x_spline(next_t)
                next_y = y_spline(next_t)
                next_yaw = np.arctan2(y_spline(next_t,1),x_spline(next_t,1))
                next_vel = v_spline(next_t)

                local_frame = vehicle_coordinate_transformation([next_x,next_y,next_yaw,next_vel],current_state)
                local_frame = local_frame[1]

                x_ref = casadi.MX([local_frame[0],local_frame[1],local_frame[2],local_frame[3]])
                cost += (x_ref.T - x[:, t].T) @ Q @ (x_ref - x[:, t])

                current_speed = None

                ref_t_list.append(next_t)
            else:
                curr_t = next_t
                ds = v_spline(curr_t)*dt
                next_t = (curr_t+ds)%track_length
                next_x = x_spline(next_t)
                next_y = y_spline(next_t)
                next_yaw = np.arctan2(y_spline(next_t,1),x_spline(next_t,1))
                next_vel = v_spline(next_t)

                local_frame = vehicle_coordinate_transformation([next_x,next_y,next_yaw,next_vel],current_state)
                local_frame = local_frame[1]

                x_ref = casadi.MX([local_frame[0],local_frame[1],local_frame[2],local_frame[3]])
                cost += (x_ref.T - x[:, t].T) @ Q @ (x_ref - x[:, t])

                ref_t_list.append(next_t)
                

        opti.subject_to(x[0, t + 1] == x[0, t] + x[3, t]*casadi.cos(x[2, t])*dt)
        opti.subject_to(x[1, t + 1] == x[1, t] + x[3, t]*casadi.sin(x[2, t])*dt)
        opti.subject_to(x[2, t + 1] == x[2, t] + x[3, t]*casadi.tan(u[1, t])*dt/WB)
        opti.subject_to(x[3, t + 1] == x[3, t] + u[0, t]*dt)

        if t < N-2:
            opti.subject_to(u[1, t+1] - u[1, t] >= -u_steer)
            opti.subject_to(u[1, t+1] - u[1, t] <= u_steer)
            opti.subject_to(u[0, t+1] - u[0, t] <= u_acc)
            opti.subject_to(u[0, t+1] - u[0, t] >= -u_acc)


    opti.subject_to(x[:, 0] == x_0)
    opti.subject_to(u[1, :] <= max_steer)
    opti.subject_to(u[1, :] >= -max_steer)
    opti.subject_to(u[0, :] <= max_acc)
    opti.subject_to(u[0, :] >= -max_acc)

    opti.minimize(cost)
    opti.solver('ipopt',{"print_time": False}, {"print_level": 0})#, {"acceptable_tol": 0.0001}
    sol = opti.solve()

    acceleration = sol.value(u[0,0])
    steering = sol.value(u[1,0])

    return acceleration, steering,ref_t_list,init_t



def rviz_markers(pose,idx):
    if idx == 0:
        points = Marker()
        points.type = Marker.POINTS
        points.header.frame_id = "map"
        points.ns = "raceline"
        points.action = Marker.ADD
        points.pose.orientation.w = 1
        points.scale.x = 0.5
        points.scale.y = 0.5
        points.color.r = 0.5
        points.color.g = 0.5
        points.color.a = 1

        for i in pose:
            p = Point()
            p.x = i[0]
            p.y = i[1]
            points.points.append(p)

    elif idx == 1:
        points = Marker()
        points.type = Marker.POINTS
        points.header.frame_id = "map"
        points.ns = "spline"
        points.action = Marker.ADD
        points.pose.orientation.w = 1
        points.scale.x = 1
        points.scale.y = 1
        points.color.b = 1
        points.color.a = 1

    for i in pose:
        p = Point()
        p.x = i[0]
        p.y = i[1]
        points.points.append(p)

    return points


if __name__ == "__main__":
    rospy.init_node("nmpc_node",anonymous=True)
    rospy.loginfo("Start NMPC")
    drive_pub = rospy.Publisher(rosparam.get_param("drive_topic"), AckermannDrive, queue_size=1)
    raceline_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)
    spline_marker_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)

    

    rate = rospy.Rate(rosparam.get_param("rate"))
    vehicle_state = VehicleState()
    # N = 5

    ref_list = np.array(global_path[:,0:2])

    init_time = rospy.get_time()
    delta_t = 0
    
    speeds = []
    desired_speeds = []

    last_t_m = 0
    
    while not rospy.is_shutdown():
        try:
            curr_t = 0
            current_state = vehicle_state.vehicle_state_output()

            # Compute Control Output from Nonlinear Model Predictive Control
            acceleration, steering,refs,t = nonlinear_kinematic_mpc_solver(current_state,last_t_m)
            last_t_m = t
            
            references = []
            for i in refs:
                references.append([x_spline(i),y_spline(i)])

            # print("Acceleration:",acceleration)
            # print("Steering:",steering)
            # print("References:",references)
                
            speeds.append(current_state[3])
            desired_speeds.append(v_spline(t))
            
            drive_msg = AckermannDrive()
            drive_msg.speed = current_state[3]+acceleration*dt
            drive_msg.steering_angle = steering
            drive_pub.publish(drive_msg)


            raceline_pub.publish(rviz_markers(global_path,0))
            spline_marker_pub.publish(rviz_markers(references,1))
            rate.sleep()
        
        except rospy.exceptions.ROSInterruptException:
            break


plt.plot(speeds,label="Actual Speed")
plt.plot(desired_speeds,label="Desired Speed")
plt.legend()
plt.show()
    