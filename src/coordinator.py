#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool,Float32MultiArray
from nav_msgs.msg import Odometry
import os
import rospkg
import rosparam
from slave import rviz_marker
from visualization_msgs.msg import Marker
rp = rospkg.RosPack()
from torch.utils.tensorboard import SummaryWriter
from subprocess import call
import time

class Coordinator:
    def __init__(self) -> None:
        self.num_cars = 3

        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=1)

        self.writer = SummaryWriter()

        self.reward = [[0]*4]*self.num_cars

        for i in range(self.num_cars):
            # rospy.Subscriber(f"/car_{i+1}/halt", Bool, self.halt_callback, callback_args=i)
            rospy.Subscriber(f"/car_{i+1}/halt", Bool, self.halt_callback)
            rospy.Subscriber(f"/car_{i+1}/done", Bool, self.done_callback, callback_args=i)
            rospy.Subscriber(f"/car_{i+1}/reward", Float32MultiArray, self.reward_callback, callback_args=i)
            rospy.Subscriber(f"/car_{i+1}/odom", Odometry, self.odom_check, callback_args=i)

        self.can_go_dict = {}
        for i in range(self.num_cars):
            # rospy.Publisher(f"/car_{i+1}/can_go", Bool, queue_size=1)
            self.can_go_dict[f"car_{i+1}"] = rospy.Publisher(f"/car_{i+1}/go", Bool, queue_size=1)

        self.run_settings_pub = rospy.Publisher("/run_settings", Float32MultiArray, queue_size=1)

        self.epsilon = 1
        min_epsilon = 0.05
        max_epsilon = 1
        self.epochs = 30000
        self.decay = (min_epsilon/max_epsilon)**(1/self.epochs)

        self.halt_flag = False

        self.start_listener = rospy.Subscriber("/start", Bool, self.start_callback)

        self.start_flag = False

        self.dones = [False]*self.num_cars

        self.all_cars_present = [False]*self.num_cars

        import pathgen_ds as pathgen
        track_file = os.path.join(rp.get_path('f1rl'),rosparam.get_param("track_file"))
        x_idx = 0
        y_idx = 1
        scale = 0.25
        self.global_path,self.track_length,self.x_spline,self.y_spline = pathgen.get_scaled_spline_path(track_file,x_idx,y_idx,scale)

    def start_callback(self, msg):
        if msg.data:
            self.start_flag = True

    def reward_callback(self, msg, car_id):
        self.reward[car_id-1] = msg.data

    
    def halt_callback(self, msg):
        if msg.data:
            self.halt_flag = True

    def done_callback(self, msg, car_id):
        self.dones[car_id-1] = msg.data

    def odom_check(self, msg, car_id):
        if msg:
            self.all_cars_present[car_id-1] = True
    

    def run(self):
        rospy.loginfo("Starting coordinator")
        while not rospy.is_shutdown():
            try:
                
                # if not self.start_flag:
                #     continue
                current_epoch = 1
                while current_epoch < self.epochs:

                    m = rviz_marker(self.global_path,0)
                    self.marker_pub.publish(m)

                    if self.halt_flag:
                        rospy.loginfo("Halt flag triggered")
                        call(["bash", os.path.join(rp.get_path('f1rl'),"src","macro_marl.sh")])
                        time.sleep(2)
                        self.halt_flag = False
                        self.all_cars_present = [False]*self.num_cars   

                    if not all(self.all_cars_present):
                        continue

                    rospy.loginfo(f"Epoch {current_epoch}")
                    settings = Float32MultiArray()
                    settings.data = [current_epoch, self.epsilon, 1 if current_epoch%50==0 else 0]

                    while not all(self.dones):
                        self.run_settings_pub.publish(settings)
                        for i in range(self.num_cars):
                            can_go_msg = Bool()
                            can_go_msg.data = True
                            self.can_go_dict[f"car_{i+1}"].publish(can_go_msg)
                            self.can_go_dict[f"car_{i+1}"].publish(can_go_msg)
                        pass

                    self.dones = [False]*self.num_cars
                    self.epsilon *= self.decay
                    current_epoch += 1

                    reward_dict = {}
                    for i in range(self.num_cars):
                        reward_dict[f"car_{i+1}"] = self.reward[i][0]

                    self.writer.add_scalars("Reward", reward_dict, current_epoch)

                    coverage_dict = {}
                    for i in range(self.num_cars):
                        coverage_dict[f"car_{i+1}"] = self.reward[i][-1]

                    self.writer.add_scalars("Coverage", coverage_dict, current_epoch)

                    rospy.loginfo(f"Epoch {current_epoch} done")

                    

            except rospy.ROSInterruptException:
                pass


if __name__ == "__main__":
    rospy.init_node("coordinator", anonymous=True)
    coordinator = Coordinator()
    coordinator.run()
    rospy.spin()


