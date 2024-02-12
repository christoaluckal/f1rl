#!/usr/bin/env python3


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
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler
import pickle
import shutil

experience_buffer = deque(maxlen=int(1e6))
BUFFER_SAMPLE = 256
reset_flag = False

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
    def __init__(self) -> None:
        rospy.init_node('car_core',anonymous=True)
        self.drive_pub = rospy.Publisher(rosparam.get_param("drive_topic"), AckermannDrive, queue_size=1)
        self.rviz_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)
        # self.spline_marker_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)
        self.reset_pub = rospy.Publisher(rosparam.get_param("reset_topic"),Pose,queue_size=1)
        self.vehicle_state = self.VehicleState()
        self.rate = rospy.Rate(rospy.get_param("rate",24))

        self.scene_reset_pub = rospy.Publisher("/scenereset",Bool,queue_size=1)
        self.car_spawner_pub = rospy.Publisher("/addcar",Bool,queue_size=1)

        track_file = os.path.join(rp.get_path('f1rl'),rosparam.get_param("track_file"))
        csv_f = track_file
        x_idx = 0
        y_idx = 1
        scale = 0.25
        self.global_path,self.track_length,self.x_spline,self.y_spline = pathgen.get_scaled_spline_path(csv_f,x_idx,y_idx,scale)


        speeds = np.arange(1,4.1,0.2)
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

            
    
    def scene_reset(self):
        print("Resetting Scene")
        self.scene_reset_pub.publish(Bool(True))
        self.rate.sleep()

    def car_spawner(self):
        # print("Spawning Car")
        self.car_spawner_pub.publish(Bool(True))
        self.rate.sleep()

    def reset(self,eps=None):
        counter = 0
        self.timesteps = 0
        self.spline_coverage = 0
        self.trajectory = []
        self.slow_counter = 0
        # random_reset = rospy.get_param("random_reset",False)
        choice = np.random.uniform(0,1)
        if choice<eps:
            frac_start = 0
            frac_end = eps
            # rand_point = np.random.randint(0,int(self.global_path.shape[0]*frac))
            start_idx = int(self.global_path.shape[0]*frac_start)
            end_idx = int(self.global_path.shape[0]*frac_end)
            rand_point = np.random.randint(start_idx,end_idx)
            next_point = rand_point+5 if rand_point<self.global_path.shape[0]-5 else 0

            x1,y1 = self.global_path[rand_point,0],self.global_path[rand_point,1]

            x1 += np.random.uniform(-1,1)
            y1 += np.random.uniform(-1,1)

            yaw = math.atan2(self.global_path[next_point,1]-y1,self.global_path[next_point,0]-x1)

            yaw += np.random.uniform(-0.1,0.1)

            # print(f"Resetting to {x1},{y1},{yaw}")

            quat = quaternion_from_euler(0,0,yaw)
        else:
            x1,y1 = 0,0
            yaw = 0
            quat = quaternion_from_euler(0,0,yaw)

        while(counter<2):
            pose = Pose()
            pose.position.x = x1
            pose.position.y = y1
            pose.position.z = 0
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

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
        
        self.spline_start = spline_t[0]

        state = [current_state[2],current_state[3]]
        spline_params = np.arange(spline_t,spline_t+3.1,0.2)

        for t in spline_params:
            state.append(self.x_spline(t)-current_state[0])
            state.append(self.y_spline(t)-current_state[1])

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

        goal_states = np.array(goal_states)

        rot_mat = np.eye(2)
        # rot_mat = np.array([[math.cos(current_state[2]),-math.sin(current_state[2])],[math.sin(current_state[2]),math.cos(current_state[2])]])

        goal_states = np.dot(goal_states,rot_mat)

        state = np.insert(goal_states.flatten(),0,current_state[3])
        state = np.insert(state,0,current_state[2])

        marker = rviz_marker([[current_state[0],current_state[1]]],1)
        self.rviz_pub.publish(marker)

        marker = rviz_marker(p,2)
        self.rviz_pub.publish(marker)


        spline_dist = self.euclidean_dist(current_state[0:2],closest_spline_point)


        self.timesteps+=1
        # move_dist = self.euclidean_dist(current_state[0:2],prev_state[0:2])

        move_dist = (closest_spline_t[0]-prev_spline_t[0])

        if abs(move_dist)>10:
            if closest_spline_t[0]>prev_spline_t[0]:
                move_dist = self.track_length-closest_spline_t[0]+prev_spline_t[0]
            else:
                move_dist = self.track_length-prev_spline_t[0]+closest_spline_t[0]

        # self.spline_coverage = (closest_spline_t[0]-self.spline_start)
                
        if self.spline_start > closest_spline_t[0]+1:
            self.spline_coverage = self.track_length+closest_spline_t[0]-self.spline_start
        else:
            self.spline_coverage = closest_spline_t[0]-self.spline_start

        if current_state[3]<0.1:
            reward=-1
            self.slow_counter+=1

            if self.slow_counter > 100:
                valid = False
                done = True

        else:

            # reward += move_dist*current_state[3] - (4-current_state[3])*1e-2

            # if spline_dist<1:
            #     reward += 10*1e-1

            # elif spline_dist<2:
            #     reward += 5*1e-1

            reward += move_dist*current_state[3]*10

            if spline_dist>1:
                print(f"Off-Track. Covered:{self.spline_coverage}/{self.track_length}")
                done=True
                reward=-100

            if self.spline_coverage > self.track_length*0.9:
                print("Track Covered")
                done=True
                reward+=100

            if len(self.trajectory)>100:
                temp = np.array(self.trajectory)
                if abs(temp[-100,0]-temp[-1,0])<0.1 and abs(temp[-100,1]-temp[-1,1])<0.1:
                    print("Step-Stalled")
                    done=True
                    reward-=50

            # if self.spline_coverage<-10:
            #     print("Backwards")
            #     done=True
            #     reward-=20

        return state,reward,done,valid,[current_state[0],current_state[1]]
        
def build_logging(experiment_name):
    root_dir = os.path.join(rp.get_path('f1rl'),'src')

    if os.path.exists(os.path.join(root_dir,'runs')):
        print("Runs Directory Exists")
    else:
        os.makedirs(os.path.join(root_dir,'runs'))

    runs_dir = os.path.join(root_dir,'runs')

    if os.path.exists(os.path.join(runs_dir,experiment_name)):
        print(f"{experiment_name} Exists")
        for f in os.listdir(os.path.join(runs_dir,experiment_name)):
            try:
                os.remove(os.path.join(runs_dir,experiment_name,f))
            except IsADirectoryError:
                shutil.rmtree(os.path.join(runs_dir,experiment_name,f))
    else:
        os.makedirs(os.path.join(runs_dir,experiment_name))

    exp_dir = os.path.join(runs_dir,experiment_name)

    if os.path.exists(os.path.join(exp_dir,'models')):
        print("Models Directory Exists")
        for f in os.listdir(os.path.join(exp_dir,'models')):
            try:
                os.remove(os.path.join(exp_dir,'models',f))
            except IsADirectoryError:
                shutil.rmtree(os.path.join(exp_dir,'models',f))

    else:
        os.makedirs(os.path.join(exp_dir,'models'))

    
    if os.path.exists(os.path.join(exp_dir,'logs')):
        print("Logs Directory Exists")
        for f in os.listdir(os.path.join(exp_dir,'logs')):
            try:
                os.remove(os.path.join(exp_dir,'logs',f))
            except IsADirectoryError:
                shutil.rmtree(os.path.join(exp_dir,'logs',f))

    else:
        os.makedirs(os.path.join(exp_dir,'logs'))


    if os.path.exists(os.path.join(runs_dir,'tensorboard',experiment_name)):
        print("Tensorboard Directory Exists")
        for f in os.listdir(os.path.join(runs_dir,'tensorboard',experiment_name)):
            os.remove(os.path.join(runs_dir,'tensorboard',experiment_name,f))

    else:
        os.makedirs(os.path.join(runs_dir,'tensorboard',experiment_name))

    
    return exp_dir
    



def create_files(experiment_path):
    print(f"Creating Files for {experiment_path}")
    with open(f"{experiment_path}/logs/dqn_reward.txt",'w') as f:
        f.write("Epoch\tReward\n")

    with open(f"{experiment_path}/logs/best_reward.txt",'w') as f:
        f.write("Epoch\tReward\n")

    with open(f"{experiment_path}/logs/trajectories.pkl",'wb') as f:
        pickle.dump({},f)

    reward_path = f"{experiment_path}/logs/dqn_reward.txt"
    best_reward_path = f"{experiment_path}/logs/best_reward.txt"
    trajectory_path = f"{experiment_path}/logs/trajectories.pkl"

    return [reward_path,best_reward_path,trajectory_path]



def shutdown_hook():
    print("Shutting Down")
    core.drive_pub.publish(AckermannDrive())
    rospy.sleep(1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architectures = [[64,64],[128,128],[256,256],[512,512],[1024,1024]]
    is_lab = rospy.get_param("is_lab",False)
    macro_file = os.path.join(rp.get_path('f1rl'),'src/macro.sh' if (not is_lab) else 'src/macro_lab.sh')

    while not rospy.is_shutdown():
        for a in architectures:

            core = CoreCarEnv()
            done=False
            ref_list = np.array(core.global_path[:,0:2])
            max_time = 100
            epochs = 10000
            max_epsilon = 1
            min_epsilon = 0.05
            decay_rate = (min_epsilon/max_epsilon)**(1/epochs)
            epsilon = max_epsilon
            
            archs = a

            architecture_str = ""
            for i in range(len(archs)):
                architecture_str+=str(archs[i])
                if i!=len(archs)-1:
                    architecture_str+="_"

            experiment_name = f"arch_{architecture_str}"
            
            exp_dir = build_logging(experiment_name)


            writer = SummaryWriter(os.path.join(exp_dir,'../tensorboard',experiment_name))

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


            trajectory = {}
            # np.save(os.path.join(rp.get_path('f1rl'),f'src/runs/path.npy'),core.global_path[:,0:2])
            np.save(os.path.join(exp_dir,'logs','path.npy'),core.global_path[:,0:2])

            reward_path,best_path,traj_path = create_files(exp_dir)


            # core.scene_reset()
            # time.sleep(1)
            # core.car_spawner()
            call(["bash",macro_file])
            print("Macro Called")
            time.sleep(1)

            epoch_counter = 0
            last_valid_epoch = 1
            current_epoch = 1
            invalid_flag = False

            rolling_rewards = []
            rolling_eval_rewards = []
            all_rewards = []

        
            path_array = np.array(core.global_path[:,0:2])
            m = rviz_marker(path_array,0)
            # core.rviz_pub.publish(m)
            try:
                while current_epoch<epochs:
                    core.rviz_pub.publish(m)
                    print(f"Epoch:{current_epoch}")
                    if current_epoch%25==0:
                        eval_mode = True
                    else:
                        eval_mode = False
                    
                    try:
                        state = core.reset(epsilon)
                        
                        rospy.sleep(1)

                        done=False
                        valid = True
                        start = time.time()
                        ep_reward=0
                        greeds = 0
                        exploits = 0
                        poses = []
                        current_trajectory = []
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

                                next_state,reward,done,valid, pos = core.step(action_idx)

                                
                                if reward != -1 and not eval_mode:
                                    experience_buffer.append((current_state,action_idx,reward,next_state,done))

                                if current_T==T:
                                    online_network = trainer.train(online_network,target_network,experience_buffer,discount_rate,BUFFER_SAMPLE,rate,device)
                                    current_T=0
                                else:
                                    current_T+=1

                                state = next_state
                                ep_reward+=reward

                                current_trajectory.append(pos)

                                if done and not valid:
                                    invalid_flag = True
                                    print("Invalid Trajectory")
                                    break
                                    

                        if invalid_flag:
                            invalid_flag = False
                            call(["bash",macro_file])
                            print("Macro Called")
                            time.sleep(1)
                            continue
                        
                        trajectory[current_epoch] = np.array(current_trajectory)

                        writer.add_scalar('Reward',ep_reward,current_epoch)
                        writer.add_scalar('Epsilon',epsilon,current_epoch)
                        if eval_mode:
                            writer.add_scalar('Eval Reward',ep_reward,current_epoch)

                        epsilon *= decay_rate
                        episodes.append(current_epoch)
                        reward_list = np.append(reward_list,ep_reward)
                        rolling_rewards.append(ep_reward)
                        all_rewards.append(ep_reward)

                        if eval_mode:
                            print(f"Eval Reward:{ep_reward}")
                            rolling_eval_rewards.append(ep_reward)
                            if ep_reward>best_reward:
                                best_reward = ep_reward
                                print(f"Best Reward:{best_reward}")
                                torch.save(online_network.state_dict(),os.path.join(exp_dir,'models',f'best_online.pth'))
                                torch.save(target_network.state_dict(),os.path.join(exp_dir,'models',f'best_target.pth'))

                        else:
                            print(f"Epoch:{current_epoch} {greeds}/{exploits} Reward:{ep_reward} Epsilon:{epsilon}")

                        writer.add_scalar('Coverage',core.spline_coverage,current_epoch)
                    
                        with open(reward_path,'a') as f:
                            f.write(f"{current_epoch}\t{ep_reward}\n")
                            
                        if current_C==C:
                            target_network.load_state_dict(online_network.state_dict())
                            current_C=0
                        else:
                            current_C+=1
                        
                            with open(best_path,'a+') as f:
                                f.write(f"{current_epoch}\t{ep_reward}\n")

                        
                        if len(rolling_rewards)==10:
                            means = np.mean(rolling_rewards)
                            writer.add_scalar('Rolling Reward',means,current_epoch)
                            rolling_rewards = []

                        if len(rolling_eval_rewards)==10:
                            means = np.mean(rolling_eval_rewards)
                            writer.add_scalar('Rolling Eval Reward',means,current_epoch)
                            rolling_eval_rewards = []

                        if current_epoch>epochs//2:
                            if np.mean(all_rewards[-100:])<0:
                                print("Early Stopping")
                                break

                        with open(traj_path,'wb') as f:
                            pickle.dump(trajectory,f)

                        current_epoch+=1

                    except (KeyboardInterrupt,rospy.ROSInterruptException) as e:
                        print(e)
                        # raise Exception("Shutting Down")
                    
            except Exception as e:
                print(e)
                break

            break



    