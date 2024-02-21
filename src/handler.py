import rospy
from geometry_msgs.msg import Pose

class CarHandler:
    def __init__(self,car_count) -> None:
        reset_pubs = {}
        listeners = {}
        for i in range(car_count):
            key = f'car_{i+1}'
            reset_pubs[key] = rospy.Publisher(f'/{key}/reset', Pose, queue_size=1)
            listeners[key] = rospy.Subscriber(f'/{key}/end', Pose, self.pose_callback)
            