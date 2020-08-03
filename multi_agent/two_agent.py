# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import habitat_sim
import os

cv2 = None

print("Current Working Directory" + os.getcwd())


# Helper function to render observations from the stereo agent
def _render(sim, display, depth=False):
    for _ in range(100):
        # Just spin in a circle
        obs = sim.step("turn_right")
        # Put the two stereo observations next to eachother
        stereo_pair_rgb = np.concatenate([obs["left_sensor"], obs["right_sensor"]], axis=1)
        stereo_pair_depth = np.concatenate([obs["left_depth_sensor"], obs["right__depth_sensor"]], axis=1)
        # If it is a depth pair, manually normalize into [0, 1]
        # so that images are always consistent
        stereo_pair_depth = np.clip(stereo_pair_depth, 0, 10)
        stereo_pair_depth /= 10.0

        # If in RGB/RGBA format, change first to RGB and change to BGR
        if len(stereo_pair_rgb.shape) > 2:
            stereo_pair_rgb = stereo_pair_rgb[..., 0:3][..., ::-1]

        # display=False is used for the smoke test
        if display:
            cv2.imshow("stereo_pair", stereo_pair_rgb)
            k = cv2.waitKey()
            if k == ord("q"):
                break


def main(display=True):
    global cv2
    # Only import cv2 if we are doing to display
    if display:
        import cv2 as _cv2
        cv2 = _cv2
        cv2.namedWindow("stereo_pair")

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene.id = (
        "data_files/skokloster-castle.glb"
    )

    # First, let's create a stereo RGB agent
    left_rgb_sensor = habitat_sim.SensorSpec()
    # Give it the uuid of left_sensor, this will also be how we
    # index the observations to retrieve the rendering from this sensor
    left_rgb_sensor.uuid = "left_sensor"
    left_rgb_sensor.resolution = [512, 512]
    # The left RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the left of the center of the agent
    left_rgb_sensor.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.LEFT
    print("left_rgb_sensor position = {}".format(left_rgb_sensor.position))
    left_rgb_sensor.orientation = [0.0, 0.0, 0.0]
    print("Left_rgb_sensor orientation = {}".format(left_rgb_sensor.orientation))
    # Same deal with the right sensor
    right_rgb_sensor = habitat_sim.SensorSpec()
    right_rgb_sensor.uuid = "right_sensor"
    right_rgb_sensor.resolution = [512, 512]
    # The right RGB sensor will be 1.5 meters off the ground
    # and 0.25 meters to the right of the center of the agent
    right_rgb_sensor.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.RIGHT
    print("right_rgb_sensor position = {}".format(right_rgb_sensor.position))
    #print("right_rgb_sensor orientation = {}".format(right_rgb_sensor.orientation))
    right_rgb_sensor.orientation = [ 0.0,0.0,0.0]
    print("right_rgb_sensor orientation = {}".format(right_rgb_sensor.orientation))


    agent1_config = habitat_sim.AgentConfiguration()

    # Now we simply set the agent's list of sensor specs to be the two specs for our two sensors
    agent1_config.sensor_specifications = [left_rgb_sensor, right_rgb_sensor]

    # Now let's do the exact same thing but for a depth camera stereo pair!
    left_depth_sensor = habitat_sim.SensorSpec()
    left_depth_sensor.uuid = "left_depth_sensor"
    left_depth_sensor.resolution = [512, 512]
    left_depth_sensor.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.LEFT
    # The only difference is that we set the sensor type to DEPTH
    left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    right_depth_sensor = habitat_sim.SensorSpec()
    right_depth_sensor.uuid = "right_depth_sensor"
    right_depth_sensor.resolution = [512, 512]
    right_depth_sensor.position = (
            1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.RIGHT
    )
    # The only difference is that we set the sensor type to DEPTH
    right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    agent2_config = habitat_sim.AgentConfiguration()
    agent2_config.sensor_specifications = [left_depth_sensor, right_depth_sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent1_config, agent2_config]))

    _render(sim, display)
    sim.close()



if __name__ == "__main__":
    main(display=True)
