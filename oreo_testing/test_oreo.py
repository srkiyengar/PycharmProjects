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

    while (1):
        #robot.UpdManCtrl()
        #robot.generate_actuator_positions(3, 4)
        robot.UpdManCtrl_new()
        keys = robot.GetKeyEvents()
        if 'c' in keys:
            robot.CheckAllCollisions()
        if 'p' in keys:
            robot.GetLinkPosOrn('neck_joint')
        if 'q' in keys:
            # quit
            break
        robot.final_pose()

    robot.Cleanup()
