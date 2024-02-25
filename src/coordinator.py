#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool,Float32MultiArray

class Coordinator:
    def __init__(self) -> None:
        self.num_cars = 2

        for i in range(self.num_cars):
            # rospy.Subscriber(f"/car_{i+1}/halt", Bool, self.halt_callback, callback_args=i)
            rospy.Subscriber(f"/car_{i+1}/halt", Bool, self.halt_callback)
            rospy.Subscriber(f"/car_{i+1}/done", Bool, self.done_callback, callback_args=i)

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

        self.dones = [False]*self.num_cars

    def start_callback(self, msg):
        if msg.data:
            self.run()

    
    def halt_callback(self, msg):
        if msg.data:
            self.halt_flag = True

    def done_callback(self, msg, car_id):
        self.dones[car_id-1] = msg.data

    

    def run(self):
        rospy.loginfo("Starting coordinator")
        while not rospy.is_shutdown():
            try:
                if self.halt_flag:
                    print("Halt flag is true")

                current_epoch = 1
                while current_epoch < self.epochs:

                    if self.halt_flag:
                        print("Halt flag is true")
                        continue

                    rospy.loginfo(f"Epoch {current_epoch}")
                    settings = Float32MultiArray()
                    settings.data = [current_epoch, self.epsilon, 1 if current_epoch%50==0 else 0]

                    self.run_settings_pub.publish(settings)

                    for i in range(self.num_cars):
                        can_go_msg = Bool()
                        can_go_msg.data = True
                        self.can_go_dict[f"car_{i+1}"].publish(can_go_msg)
                        self.can_go_dict[f"car_{i+1}"].publish(can_go_msg)

                    while not all(self.dones):
                        pass

                    self.dones = [False]*self.num_cars
                    self.epsilon *= self.decay
                    current_epoch += 1

                    

            except rospy.ROSInterruptException:
                pass


if __name__ == "__main__":
    rospy.init_node("coordinator", anonymous=True)
    coordinator = Coordinator()
    coordinator.run()
    rospy.spin()


