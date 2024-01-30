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
from subprocess import Popen,call
from torch.utils.tensorboard import SummaryWriter

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

        spline_dist = self.euclidean_dist(current_state[0:2],closest_spline_point)

        

        current_rot = self.vehicle_state.get_state()[1]

        
        # if math.isclose(current_rot[1],-1,abs_tol=0.1):
        #     print("Flipped")
        #     done=True
        #     reward-=100

        # if self.euclidean_dist(current_state[0:2],prev_state[0:2])<1e-3 and self.timesteps>200:
        #     print("Stuck")
        #     done=True
        #     reward-=100

        self.timesteps+=1
        # move_dist = self.euclidean_dist(current_state[0:2],prev_state[0:2])
        move_dist = (closest_spline_t[0]-prev_spline_t[0])

        # print(f"Move Dist:{move_dist}")
        if current_state[3]<0.25:
            reward-=1
        
        else:

            reward += move_dist*current_state[3]*10

            if spline_dist>3:
                print("Off Track")
                done=True
                reward=-1

            if closest_spline_t>60:
                print("Finished Track")
                done=True
                reward+=10

        return state,reward,done
        



def shutdown_hook():
    print("Shutting Down")
    core.drive_pub.publish(AckermannDrive())
    rospy.sleep(1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architectures = [[128,128],[128,128,128],[256,256],[256,256,256],[512,512],[512,512,512]]
    is_lab = rospy.get_param("is_lab",False)
    macro_file = os.path.join(rp.get_path('f1rl'),'src/macro.sh' if (not is_lab) else 'src/macro_lab.sh')

    exp_counter = 0

    for a in architectures:
        exp_counter+=1
        core = CoreCarEnv()
        done=False
        ref_list = np.array(core.global_path[:,0:2])
        max_time = 100
        epochs = 3000
        max_epsilon = 1
        min_epsilon = 0.01
        decay_rate = (min_epsilon/max_epsilon)**(1/epochs)
        epsilon = max_epsilon
        
        archs = a

        experiment_name = f"experiment_{exp_counter}"

        writer = SummaryWriter(os.path.join(rp.get_path('f1rl'),'src/tensorboard',experiment_name))

        n_states = core.get_state_count()

        episodes = []
        reward_list = np.array([])
        mean_list = np.array([])


        online_network = Network(n_states,core.action_count,archs).to(device)
        target_network = Network(n_states,core.action_count,archs).to(device)

        target_network.load_state_dict(online_network.state_dict())

        best_reward = -1e6

        discount_rate = 0.99
        rate = 1e-3
        policy = EpsilonGreedy.policy
        trainer = NaiveDQN()

        C = 20
        T = 20
        
        current_C = 0
        current_T = 0
        rospy.on_shutdown(shutdown_hook)

        eval_mode = False

        call(["bash",macro_file])
        print("Macro Called")
        time.sleep(1)
        
        while not rospy.is_shutdown():
            try:
                for i in range(epochs):
                    if i%50==0:
                        eval_mode = True
                    else:
                        eval_mode = False
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
                                if eval_mode:
                                    action_idx,action_type = policy(state,online_network,core.action_count,-1,device)
                                else:
                                    action_idx,action_type = policy(state,online_network,core.action_count,epsilon,device)

                                if action_type==0:
                                    greeds+=1
                                else:
                                    exploits+=1

                                next_state,reward,done = core.step(action_idx)

                                experience_buffer.append((current_state,action_idx,reward,next_state,done))

                                if current_T==T:
                                    online_network = trainer.train(online_network,target_network,experience_buffer,discount_rate,BUFFER_SAMPLE,rate,device)
                                    current_T=0
                                else:
                                    current_T+=1
                                state = next_state
                                ep_reward+=reward

                        writer.add_scalar('Reward',ep_reward,i)
                        if eval_mode:
                            writer.add_scalar('Eval Reward',ep_reward,i)

                        epsilon *= decay_rate
                        episodes.append(i)
                        reward_list = np.append(reward_list,ep_reward)

                        if eval_mode:
                            print(f"Eval Reward:{ep_reward}")
                        print(f"Epoch:{i} {greeds}/{exploits} Reward:{ep_reward} Epsilon:{epsilon}")
                    
                        with open(os.path.join(rp.get_path('f1rl'),'src/dqn_reward.txt'),'a') as f:
                            f.write(f"{i}\t{ep_reward}\n")
                            
                        if current_C==C:
                            target_network.load_state_dict(online_network.state_dict())
                            current_C=0
                        else:
                            current_C+=1

                        if ep_reward>0 or i%50==49:
                            torch.save(online_network.state_dict(),os.path.join(rp.get_path('f1rl'),'src/online_network.pth'))
                            torch.save(target_network.state_dict(),os.path.join(rp.get_path('f1rl'),'src/target_network.pth'))
                            print("Saved Network")

                        

                            with open(os.path.join(rp.get_path('f1rl'),'src/best_reward.txt'),'a') as f:
                                f.write(f"{i}\t{ep_reward}\n")

                        means = np.mean(reward_list[-10:]) if len(reward_list)>10 else None
                        print(f"Mean Reward:{means} Best Reward:{best_reward}")

                        if (means is not None and means>-300 and means<-200) or i%25==24:
                            call(["bash",macro_file])
                            print("Macro Called")
                            time.sleep(1)

                        if (means is not None) and (means>best_reward):
                            print(f"Best Mean Reward:{means}")
                            torch.save(online_network.state_dict(),os.path.join(rp.get_path('f1rl'),'src/online_network_best.pth'))
                            torch.save(target_network.state_dict(),os.path.join(rp.get_path('f1rl'),'src/target_network_best.pth'))
                            print("Saved Best Network")
                            best_reward = means

                    except (KeyboardInterrupt,rospy.ROSInterruptException) as e:
                        print(e)
                        # raise Exception("Shutting Down")
                    
            except Exception as e:
                print(e)
                break

            break



    