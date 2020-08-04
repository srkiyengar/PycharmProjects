import pybullet as p
import time as t
import pybullet_data
from scipy.spatial.transform import Rotation as R
import logging
import oreo
import numpy as np

if __name__ == "__main__":
    robot = oreo.Oreo_Robot(True, True, "/home/oreo/Documents/oreo_sim/oreo/sim", "assembly.urdf", True)
    robot.InitModel()
    print(robot.GetJointNames())
    robot.InitManCtrl()
    robot.RegKeyEvent(['c', 'q', 'p'])

    a = robot.read_oreo_yaw_pitch_actuator_data()
    if a == 0:
        print("Building scan data takes minutes ....")
        robot.build_oreo_scan_yaw_pitch_actuator_data()
    robot.produce_interpolators()
    k = 0

    while (1):
        #robot.UpdManCtrl()
        #robot.UpdManCtrl_new()

        if k == 0:
            robot.look_at_point(0.50,0.4,0.50)
            k = 1

        keys = robot.GetKeyEvents()
        if 'c' in keys:
            robot.CheckAllCollisions()
        if 'p' in keys:
            robot.GetLinkPosOrn('neck_joint')
        if 'q' in keys:
            # quit
            break
        #robot.final_pose()
    robot.Cleanup()
