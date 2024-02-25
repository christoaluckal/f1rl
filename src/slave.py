#!/usr/bin/env python3


import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Point,Pose
from visualization_msgs.msg import Marker
import math
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
from collections import deque
from networks.policies import EpsilonGreedy
from networks.dqn import NaiveDQN
from networks.nns import General_Network as Network
import torch
from torch.utils.tensorboard import SummaryWriter
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String,Float32MultiArray
import sys



def distance_to_spline(t,current_x,current_y,x_spline,y_spline):
    spline_x, spline_y = x_spline(t), y_spline(t)
    return math.sqrt((spline_x - current_x) ** 2 + (spline_y - current_y) ** 2)

def rviz_marker(msg,msg_type):
    if msg_type == 0:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.a = 1.0
        for p in msg:
            point = Point()
            point.x = p[0]
            point.y = p[1]
            point.z = 0
            marker.points.append(point)
    elif msg_type == 1:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "ego"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.color.g = 1.0
        marker.color.a = 1.0
        for p in msg:
            point = Point()
            point.x = p[0]
            point.y = p[1]
            point.z = 0
            marker.points.append(point)

    elif msg_type == 2:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "reference"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color.b = 1.0
        marker.color.a = 1.0
        for p in msg:
            point = Point()
            point.x = p[0]
            point.y = p[1]
            point.z = 0
            marker.points.append(point)

    return marker

class CoreCarEnv():
    def __init__(self,idx,archs) -> None:
        
        self.idx = idx
        drive_topic = f'/car_{idx}/command'
        odom_topic = f'/car_{idx}/odom'
        reset_topic = f'/car_{idx}/reset'
        halting_topic = f'/car_{idx}/halt'
        go_topic = f'/car_{idx}/go'
        done_topic = f'/car_{idx}/done'
        run_settings = '/run_settings'
        car_adder = '/addcar'

        self.drive_pub = rospy.Publisher(drive_topic, AckermannDrive, queue_size=1)
        self.rviz_pub = rospy.Publisher('/visualization_marker',Marker,queue_size=1)
        self.reset_pub = rospy.Publisher(reset_topic,Pose,queue_size=3)
        self.halt_pub = rospy.Publisher(halting_topic,Bool,queue_size=1)
        self.go_sub = rospy.Subscriber(go_topic,Bool,self.go_callback)
        self.run_settings_sub = rospy.Subscriber(run_settings,Float32MultiArray,self.update_settings)
        self.done_pub = rospy.Publisher(done_topic,Bool,queue_size=1)
        self.addcar_pub = rospy.Publisher(car_adder,String,queue_size=1)

        self.current_settings = None
        self.requested_reset = False
        self.can_go = False

        self.vehicle_state = self.VehicleState(odom_topic)
        self.rate = rospy.Rate(rospy.get_param("rate",24))

        track_file = os.path.join(rp.get_path('f1rl'),rosparam.get_param("track_file"))
        csv_f = track_file
        x_idx = 0
        y_idx = 1
        scale = 0.25
        self.global_path,self.track_length,self.x_spline,self.y_spline = pathgen.get_scaled_spline_path(csv_f,x_idx,y_idx,scale)


        speeds = [2,3,4]
        turns = np.arange(-0.4,0.41,0.1)

        self.action_map = {}
        self.trajectory = []
        counter = 0
        for s in speeds:
            for t in turns:
                self.action_map[counter]=[s,t]
                counter+=1

        self.action_count = len(self.action_map.keys())

        self.timesteps = 0

        self.spline_coverage = 0
        self.spline_start = 0

        self.slow_counter = 0
        self.spline_offset = 0

        self.cross_over = False

        self.done=False

        architecture_str = ""
        for i in range(len(archs)):
            architecture_str+=str(archs[i])
            if i!=len(archs)-1:
                architecture_str+="_"


        n_states = self.get_state_count()

        self.episodes = []
        self.reward_list = np.array([])
        self.mean_list = np.array([])


        self.online_network = Network(n_states,self.action_count,archs).to(device)
        self.target_network = Network(n_states,self.action_count,archs).to(device)

        self.best_reward = -1e6

        self.discount_rate = 0.99
        self.lr = 1e-3
        self.policy = EpsilonGreedy.policy
        self.trainer = NaiveDQN()

        self.current_C = 0
        self.C = 20

        self.current_T = 0
        self.T = 100

    def update_model(self,msg):
        model_path = msg.data
        self.online_network.load_state_dict(torch.load(model_path))

    def go_callback(self,msg):
        self.can_go = msg.data

    def update_settings(self,msg):
        self.current_settings = msg.data
        

    class VehicleState:
        def __init__(self,topic):
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
            self.odom_sub = rospy.Subscriber(topic, Odometry, self.odom_callback)

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
        # res = minimize(fn,x0=best_t, args=(current_x, current_y,x_spline,y_spline))
        # return res.x
        skips = 25
        deltas = self.track_length/skips
        ts = np.arange(0,self.track_length,deltas)

        min_dist = 1e6
        min_t = 0

        for t in ts:
            res = minimize(fn,x0=t, args=(current_x, current_y,x_spline,y_spline),bounds=((0,self.track_length),))
            p = [x_spline(res.x),y_spline(res.x)]
            dist = self.euclidean_dist([current_x,current_y],p)
            if dist<min_dist:
                min_dist = dist
                min_t = res.x

        return min_t

    def reset(self,eps=None):
        counter = 0
        self.timesteps = 0
        self.spline_coverage = 0
        self.trajectory = []
        self.slow_counter = 0
        self.spline_offset = 0
        self.cross_over = False
        # random_reset = rospy.get_param("random_reset",False)
        choice = np.random.uniform(0,1)
        if choice>-1:
            frac_start = 0.9
            frac_end = 1
            # rand_point = np.random.randint(0,int(self.global_path.shape[0]*frac))
            start_idx = int(self.global_path.shape[0]*frac_start)
            end_idx = int(self.global_path.shape[0]*frac_end)
            rand_point = np.random.randint(start_idx,end_idx)
            next_point = rand_point+5 if rand_point<self.global_path.shape[0]-5 else 0

            x1,y1 = self.global_path[rand_point,0],self.global_path[rand_point,1]

            x1 += np.random.uniform(-1,1)
            y1 += np.random.uniform(-1,1)

            # yaw = math.atan2(self.global_path[next_point,1]-y1,self.global_path[next_point,0]-x1)

            spline_t_dash = self.closest_spline_param(distance_to_spline,x1,y1,self.x_spline,self.y_spline)
        
            dx_dt = self.x_spline(spline_t_dash,1)
            dy_dt = self.y_spline(spline_t_dash,1)

            yaw = math.atan2(dy_dt,dx_dt)
            yaw += np.random.uniform(-0.1,0.1)


            # print(f"Resetting to {x1},{y1},{yaw}")

            quat = quaternion_from_euler(0,0,yaw)
        else:
            x1,y1 = 0,0
            yaw = 0
            quat = quaternion_from_euler(0,0,yaw)

        # while(counter<5):
        pose = Pose()
        pose.position.x = x1
        pose.position.y = y1
        pose.position.z = 0
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        self.reset_pub.publish(pose)

        rospy.Rate(1).sleep()
        counter+=1

        current_state = self.vehicle_state.get_state()[0]
        # while(current_state[3]>1e-1):
        #     self.drive_pub.publish(AckermannDrive())
        #     self.rate.sleep()
        #     current_state = self.vehicle_state.get_state()[0]

        current_state = self.vehicle_state.get_state()[0]
        spline_t = self.closest_spline_param(distance_to_spline,current_state[0],current_state[1],self.x_spline,self.y_spline)
        
        self.spline_start = spline_t[0]
        print(f"Car {self.idx} Spline Start:{self.spline_start}")

        state = [current_state[2],current_state[3]]
        
        spline_params = np.arange(spline_t,spline_t+3.1,0.2)

        for t in spline_params:
            try:
                state.append(self.x_spline(t)-current_state[0])
                state.append(self.y_spline(t)-current_state[1])
            except:
                x_sp = self.x_spline(t%self.track_length)
                y_sp = self.y_spline(t%self.track_length)
                state.append(x_sp-current_state[0])
                state.append(y_sp-current_state[1])

        state = np.array(state)
        return state
    
    def get_state_count(self):
        current_state = [0,0,0,0]
        spline_t = self.closest_spline_param(distance_to_spline,current_state[0],current_state[1],self.x_spline,self.y_spline)

        state = [current_state[2],current_state[3]]
        spline_params = np.arange(spline_t,spline_t+3.1,0.2)

        for t in spline_params:
            # state.append(self.x_spline(t))
            # state.append(self.y_spline(t))
            # state.append(current_state[0]-self.x_spline(t))
            # state.append(current_state[1]-self.y_spline(t))
            state.append(self.x_spline(t)-current_state[0])
            state.append(self.y_spline(t)-current_state[1])

        state = np.array(state)
        return state.shape[0]


    def step(self,idx):
        # print(self.trajectory.shape)
        reward = 0
        done = False
        valid = True
        prev_state = self.vehicle_state.get_state()[0]
        prev_spline_t = self.closest_spline_param(distance_to_spline,prev_state[0],prev_state[1],self.x_spline,self.y_spline)

        
        ack_msg = AckermannDrive()
        action = self.action_map[idx]


        ack_msg.speed = 2
        ack_msg.steering_angle = 0

        self.drive_pub.publish(ack_msg)
        self.rate.sleep()

        current_state = self.vehicle_state.get_state()[0]
        closest_spline_t = self.closest_spline_param(distance_to_spline,current_state[0],current_state[1],self.x_spline,self.y_spline)
        spline_x, spline_y = self.x_spline(closest_spline_t[0]), self.y_spline(closest_spline_t[0])

        closest_spline_point = np.array([spline_x,spline_y])

        self.trajectory.append([spline_x,spline_y])

        spline_params = np.arange(closest_spline_t[0],closest_spline_t[0]+3.1,0.2)

        
        p = []
        goal_states = []

        for t in spline_params:
            lx = self.x_spline(t)-current_state[0]
            ly = self.y_spline(t)-current_state[1]
            # state.append(lx)
            # state.append(ly)
            goal_states.append([lx,ly])
            
            p.append([self.x_spline(t),self.y_spline(t)])

        path_yaw = math.atan2(p[-1][1]-p[0][1],p[-1][0]-p[0][0])

        goal_states = np.array(goal_states)

        rot_mat = np.eye(2)
        # rot_mat = np.array([[math.cos(current_state[2]),-math.sin(current_state[2])],[math.sin(current_state[2]),math.cos(current_state[2])]])

        goal_states = np.dot(goal_states,rot_mat)

        state = np.insert(goal_states.flatten(),0,current_state[3])
        state = np.insert(state,0,current_state[2])

        # marker = rviz_marker([[current_state[0],current_state[1]]],1)
        # self.rviz_pub.publish(marker)

        # marker = rviz_marker(p,2)
        # self.rviz_pub.publish(marker)


        spline_dist = self.euclidean_dist(current_state[0:2],closest_spline_point)


        self.timesteps+=1
        # move_dist = self.euclidean_dist(current_state[0:2],prev_state[0:2])

        move_dist = (closest_spline_t[0]-prev_spline_t[0])

        if abs(move_dist)>10:
            if closest_spline_t[0]>prev_spline_t[0]:
                move_dist = self.track_length-closest_spline_t[0]+prev_spline_t[0]
            else:
                move_dist = self.track_length-prev_spline_t[0]+closest_spline_t[0]


        if self.spline_start > closest_spline_t[0]+1:
            

            # self.spline_coverage = abs(self.spline_start-closest_spline_t[0]) + self.spline_offset
            # if abs(current_state[2]-path_yaw) > math.pi*0.8:
            #     self.spline_coverage = -1*self.spline_coverage
            self.spline_coverage = closest_spline_t[0]-self.spline_start + self.spline_offset


            # print(f"A)Spline Start:{np.round(self.spline_start,3)}\nClosest Spline:{np.round(closest_spline_t[0],3)}\nCoverage:{np.round(self.spline_coverage,3)}")
        else:
            if math.isclose(closest_spline_t[0],self.track_length,abs_tol=1):
                # print("Close to End")
                if not self.cross_over:
                    # print("Cross Over")
                    self.spline_offset = (self.track_length-self.spline_start)
                    self.spline_start = 0
                    self.cross_over = True

            if math.isclose(closest_spline_t[0],self.track_length,abs_tol=1):
                self.spline_coverage = self.spline_offset
            else:
                self.spline_coverage = closest_spline_t[0] -self.spline_start + self.spline_offset


            
            # print(f"B)Spline Start:{np.round(self.spline_start,3)}\nClosest Spline:{np.round(closest_spline_t[0],3)}\nCoverage:{np.round(self.spline_coverage,3)}")
            

        if current_state[3]<0.1:
            reward=-1
            self.slow_counter+=1

            if self.slow_counter > 100:
                valid = False
                done = True

        elif self.spline_coverage < -4:
            print(f"Backward. Covered:{self.spline_coverage}/{self.track_length}")
            reward = -10
            done = True
        else:

            reward += move_dist*(current_state[3])*5

            if spline_dist>1:
                print(f"Car {self.idx} Off-Track. Covered:{self.spline_coverage}/{self.track_length}")
                done=True
                reward=-10

            if self.spline_coverage > self.track_length*0.9 and len(self.trajectory)>100:
                print("Track Covered")
                done=True
                reward+=10

            if len(self.trajectory)>100:
                temp = np.array(self.trajectory)
                if abs(temp[-100,0]-temp[-1,0])<0.1 and abs(temp[-100,1]-temp[-1,1])<0.1:
                    print("Step-Stalled")
                    done=True
                    reward-=5

        return state,reward,done,valid,[current_state[0],current_state[1]]
    
    def run_episode(self,epoch_num,epsilon,eval_mode=False):
        while True:
            init_state = self.reset()
            try:
                rospy.wait_for_message(f'/car_{self.idx}/odom',Odometry,timeout=3)
                print(f"Car {self.idx} Odom Received")
                break
            except:
                car_name = f'car_{self.idx}'
                s = String()
                s.data = car_name
                self.addcar_pub.publish(s)
                continue

        state = init_state
        done = False
        valid = True
        ep_reward = 0
        greeds = 0
        exploits = 0
        coverage = 0
        current_trajectory = []
        start = time.time()

        current_experience = []

        while not done:
            # rospy.loginfo(f"Car {self.idx} T:{self.timesteps}")
            if eval_mode:
                action,action_type = self.policy(state,self.online_network,self.action_count,-1,device)
            else:
                action,action_type = self.policy(state,self.online_network,self.action_count,epsilon,device)
            next_state,reward,done,valid,trajectory = self.step(action)
            current_experience.append((state,action,reward,next_state,done))

            if done and not valid:
                break

            if action_type == 1:
                greeds+=1
            else:
                exploits+=1

            ep_reward+=reward
            state = next_state
            current_trajectory.append(trajectory)

            if time.time()-start > 100:
                done = True
                reward -= 100
                break

        if done and not valid:
            self.requested_reset = True
            self.halt_pub.publish(Bool(True))

    def run(self):
        rospy.loginfo(f"Car {self.idx} Running")
        while not rospy.is_shutdown():
            if self.requested_reset:
                continue
            
            if not self.can_go:
                continue
            
            if self.current_settings is None:
                continue
            
            eval_mode = True if self.current_settings[-1] == 1 else False

            epoch,epsilon = self.current_settings[0],self.current_settings[1]

            print(f"Car:{self.idx} Epoch:{epoch} Eps:{epsilon} Eval:{eval_mode}")

            self.can_go = False

            self.run_episode(epoch,epsilon,eval_mode)

            self.done_pub.publish(Bool(True))


        # self.run_episode(epoch,epsilon,eval_mode)


            

if __name__ == "__main__":
    cidx = int(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rospy.init_node(f'slave_{cidx}', anonymous=True)
    archs = [128,128]
    car = CoreCarEnv(cidx,archs)
    car.run()