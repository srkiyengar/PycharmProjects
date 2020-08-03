import habitat_sim

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import pickle

cv2 = None
print("Current Working Directory" + os.getcwd())

def generate_states(sim):
    agentStateList = []
    for _ in range(0, 100):
        agentStateList.append(habitat_sim.AgentState())
        obs = sim.step("turn_right")
    sim.agents[0].set_state(agentStateList[0])
    pickle_out = open("pickled_state","wb")
    pickle.dump(agentStateList,pickle_out)
    pickle_out.close()
    return

'''agent = sim.initialize_agent(0)
original_state = habitat_sim.AgentState()
q = sim.agents[0].state.rotation
q_pos = sim.agents[0].state.position
p = habitat_sim.AgentState().rotation'''


# Helper function to render observations from the stereo agent
def _render(sim, display):
    agent = sim.initialize_agent(0)
    initial_agent_state = habitat_sim.AgentState()
    myAgentStates = pickle.load(open("pickle_out","rb"))
    for agent_state in myAgentStates:
        # Just spin in a circle
        agent.set_state(agent_state)

        for _, sensor in sim._sensors.items():
            sensor.draw_observation()

        observations = {}
        for sensor_uuid, sensor in sim._sensors.items():
            observations[sensor_uuid] = sensor.get_observation()
        rgb_left = observations["left_sensor"]
        rgb_right = observations["right_sensor"]
        depth = observations["depth_sensor"]

        if len(rgb_left.shape) > 2:
            rgb_left = rgb_left[..., 0:3][..., ::-1]
        if len(rgb_right.shape) > 2:
            rgb_right = rgb_right[..., 0:3][..., ::-1]
        depth = np.clip(depth, 0, 10)
        depth /= 10.0

        stereo_pair = np.concatenate([rgb_left, rgb_right], axis=1)

        if display:
            cv2.imshow("stereo_pair", stereo_pair)
            k = cv2.waitKey()
            if k == ord("q"):
                break
            '''else:
                cv2.imshow("stereo_pair", depth)
                k = cv2.waitKey()
                if k == ord("q"):
                    break'''

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

    # Now let's do the exact same thing but for a depth camera stereo pair!
    depth_sensor = habitat_sim.SensorSpec()
    depth_sensor.uuid = "depth_sensor"
    depth_sensor.resolution = [512, 512]
    depth_sensor.position = [0.0, 0.0, 0.0]
    # The only difference is that we set the sensor type to DEPTH
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    agent_config = habitat_sim.AgentConfiguration()

    # Now we simply set the agent's list of sensor specs to be the two specs for our two sensors
    agent_config.sensor_specifications = [left_rgb_sensor, right_rgb_sensor, depth_sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))
    generate_states(sim)
    _render(sim, display)
    sim.close()


if __name__ == "__main__":
    main(display=True)