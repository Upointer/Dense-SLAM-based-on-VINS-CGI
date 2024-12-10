#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt

class OdomPlotter:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        rospy.init_node('odom_plotter', anonymous=True)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.on_shutdown(self.plot_trajectory)

    def odom_callback(self, msg):
        # 提取位置数据
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.x_data.append(x)
        self.y_data.append(y)

    def plot_trajectory(self):
        # 绘制轨迹
        plt.figure()
        plt.plot(self.x_data, self.y_data, label='RF2O Trajectory')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RF2O Odometry Trajectory')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    try:
        plotter = OdomPlotter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
