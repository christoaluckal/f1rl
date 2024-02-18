#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDrive
import sys
import time
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

def reset():

    car_adder.publish()
    time.sleep(1)

    car_adder.publish("car_1")
    time.sleep(2)


    car_1_pose = Pose()
    car_1_pose.position.x = 2
    car_1_pose.position.y = 0
    car_1_pose.position.z = 0

    roll = 0
    pitch = 0
    yaw = np.pi

    quat = quaternion_from_euler(roll,pitch,yaw)
    car_1_pose.orientation.x = quat[0]
    car_1_pose.orientation.y = quat[1]
    car_1_pose.orientation.z = quat[2]
    car_1_pose.orientation.w = quat[3]

    car_1_resetter.publish(car_1_pose)
    time.sleep(2)
    print("Spawned and resetted Real car_1")
    

    car_adder.publish("car_2")
    time.sleep(2)

    car_2_pose = Pose()
    car_2_pose.position.x = 0
    car_2_pose.position.y = 0
    car_2_pose.position.z = 0
    car_2_resetter.publish(car_2_pose)
    time.sleep(1)

    print("Spawned and resetted car_2")
    

    return

if __name__ == '__main__':
    rospy.init_node('unity_dev', anonymous=True)
    
    car_adder = rospy.Publisher('/addcar',String,queue_size=1)
    car_1_cmd = rospy.Publisher('/car_1/command',AckermannDrive,queue_size=1)
    car_1_resetter = rospy.Publisher('/car_1/reset',Pose,queue_size=1)
    car_2_cmd = rospy.Publisher('/car_2/command',AckermannDrive,queue_size=1)
    car_2_resetter = rospy.Publisher('/car_2/reset',Pose,queue_size=1)

    rate = rospy.Rate(10)

    


    try:
        print("Starting")
        reset()

        while True:
            if input("Press enter to continue") == "":
                break

        while not rospy.is_shutdown():   
            try:
                car_2_cmd.publish(AckermannDrive(steering_angle=0.0,speed=1.0))
                rate.sleep()
            except KeyboardInterrupt:
                car_2_cmd.publish(AckermannDrive(steering_angle=0.0,speed=0.0))
                car_2_cmd.publish(AckermannDrive(steering_angle=0.0,speed=0.0))
                break

    except Exception:
        reset()
        print("Resetted 2")
        pass

        

    

        

