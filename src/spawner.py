#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDrive
import sys
import time
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

def reset(resets):

    car_adder.publish()
    time.sleep(2)

    for i in range(4):
        print(f"Adding car_{i+1}")
        car_adder.publish(f"car_{i+1}")
        time.sleep(2)
        pose = Pose()
        pose.position.x = np.random.uniform(-5,5)
        resets[i].publish(pose)
        time.sleep(2)

if __name__ == '__main__':
    rospy.init_node('car_spawner', anonymous=True)
    
    car_adder = rospy.Publisher('/addcar',String,queue_size=1)
    resets = {}
    for i in range(4):
        resets[i] = rospy.Publisher('/car_{}/reset'.format(i+1),Pose,queue_size=1)
    

    try:
        print("Starting")
        reset(resets=resets)

    except Exception:
        reset()
        print("Resetted 2")
        pass

        

    

        

