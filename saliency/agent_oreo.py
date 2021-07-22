import habitat_sim
import numpy as np
import quaternion
import math
import cv2
import oreo
from PIL import Image
import os
import pickle
from datetime import datetime
import saliency
import matplotlib.pyplot as plt




eye_separation = 0.058
sensor_resolution = [512,512]
#sensor_resolution = [256,256]
#scene = "../multi_agent/data_files/skokloster-castle.glb"

dest_folder = "/Users/rajan/PycharmProjects/saliency/saliency_map"
scene = "../multi_agent/data_files/van-gogh-room.glb"


def homogenous_transform(R, vect):
    """
    :param R: 3x3 matrix
    :param vect: list x,y,z
    :return:Homogenous transformation 4x4 matrix using R and vect
    """

    H = np.zeros((4, 4))
    H[0:3, 0:3] = R
    frame_displacement = vect + [1]
    D = np.array(frame_displacement)
    D.shape = (1, 4)
    H[:, 3] = D
    return H

def inverse_homogenous_transform(H):
    """
    :param H: Homogenous Transform Matrix
    :return: Inverse Homegenous Transform Matrix
    """

    R = H[0:3, 0:3]
    origin = H[:-1, 3]
    origin.shape = (3, 1)

    R = R.T
    origin = -R.dot(origin)
    return homogenous_transform(R, list(origin.flatten()))

'''
Habitat coordinate system (HCS) - At zero rotation of both sensor and agent, the sensor is looking in the
direction of -ive z. Forward motion of the agent is traveling in -ve z direction.
+ive y is UP. Sensor rotations are available only as rotation wrt the habitat frame.

The Agent and Sensor coordinate frames are chosen to be aligned to HCS. For the purpose of integrating PyBullet
to Habitat, Pybullet is aligned to the Agent frame. The sensor direction will be computed with respect 
to the current Agent frame.

'''

def rotatation_matrix_from_Habitat_to_Pybullet():
    R = np.zeros((3, 3))
    R[0, 1] = -1
    R[1, 2] = 1
    R[2, 0] = -1

    return R


def rotatation_matrix_from_Pybullet_to_Habitat():
    R = np.zeros((3, 3))
    R[0, 2] = -1
    R[1, 0] = -1
    R[2, 1] = 1

    return R

'''
Habitat Pybullet rotation is only used for checking if saccade is within range.
Therefore, a given agent orientation in habitat is assumed to be aligned with the head/neck orientation in PyBullet.
'''


R_HA_to_PyB = quaternion.from_rotation_matrix(rotatation_matrix_from_Habitat_to_Pybullet())
R_PyB_to_HA = quaternion.from_rotation_matrix(rotatation_matrix_from_Pybullet_to_Habitat())


def compute_eye_saccade_from_PyBframe(eye_rot):
    """
    A given yaw,pitch of an eye in PyBullet frame wrt head/neck,
    it returns sensor rotation wrt Agent of Habitat
    :param eye_rot: in quaternion, rotation of the eye wrt to the head/neck
    :return: in quaternion, the rotation of the sensor wrt to agent frame.

    PyBullet is used to check if a saccade of the eye is within the range of Oreo can perform
    The PyBullet Head/neck frame alignment to Habitat Agent frame is provided by R_PyB_to_HA and
    it inverse R_HA_to_PyB.
    R_HA_to_PyB * eye_rot gives the orientation of the pybullet sensor wrt to Habitat Agent frame.
    The rotation R_PyB_to_HA will take out the HA to PyB rotation providing sensor rotation wrt Agent
    """

    return (R_HA_to_PyB * eye_rot) * R_PyB_to_HA


def display_image(images, left=True, right=False):

    a = len(images)
    left_img = images[0]
    right_img = images[1]

    # cv2.imshow require BGR using img = img[..., ::-1]

    left_img = left_img[..., 0:3][..., ::-1]
    right_img = right_img[..., 0:3][..., ::-1]

    if a == 2:
        if left:
            cv2.imshow("Left_eye", left_img)
            #get_image_patch(images[0],256,256,50)
        if right:
            cv2.imshow("Right_eye", right_img)
    elif a == 3:
        if left:
            cv2.imshow("Left_eye", left_img)
        if right:
            cv2.imshow("Right_eye", right_img)
        cv2.imshow("Depth", images[2]/10)
    return


def get_image_patch(source_image, xloc, yloc, size):

    patch = source_image[xloc-size:xloc+size, yloc-size:yloc+size,0:3]
    cv2.imshow(f"Patch at {xloc}, {yloc}", patch)
    pass


def save_image_as_image_file(any_image, name):
    """

    :param any_image: ndarray
    :param name: name should have some_name.png or jpg etc.
    The image in BGR is converted to RGB, then converts it to a PIL image and saved in the file format indicated
    by the name.
    :return:
    """
    new_im = any_image[...,::-1].copy()
    im = Image.fromarray(new_im)
    #im = Image.fromarray(any_image)
    im.save(name)
    return


def calculate_rotation_to_new_direction(uvector):
    """
"   Computes quaternion to rotate unit vector in -z direction (0,0,-1) to align with the uvector.
    This is the rotation of sensor from its current orientation to point to new orientation.
    :param uvector: numpy array unit vector which is the new direction where sensor will look
    :return: rotation in quaternion wrt to current sensor frame.
    """

    v1 = uvector
    v2 = np.array([0.0, 0.0, -1.0])
    # my_axis is v2 cross v1
    my_axis = np.cross(v2, v1)
    my_axis = my_axis / np.linalg.norm(my_axis)
    my_angle = np.arccos(np.dot(v1, v2))
    my_axis_angle = my_angle * my_axis
    quat = quaternion.from_rotation_vector(my_axis_angle)
    return quat

def compute_pixel_in_current_frame(R1, R2, pixels_in_previous_frame, focal_distance, width, height):
    '''
    Reference for rotation is Habitat World Coordinates
    :param R1 Rotation in quaterion from WCS to previous sensor(eye) frame
    :param R2 Rotation in quaterion from WCS to current sensor(eye) frame
    :param pixels_in_previous_frame: List of x,y positions of pixel in the previous frame
    :param focal_distance: the frame z coordinate is equal to -focal_distance
    :param width: sensor width pixels
    :param height: sensor height pixels
    :returns List of x, y positions in the current frame
    '''

    R = R2.inverse()*R1 #Rotation from current frame to previous frame
    w = width/2
    h = height/2
    new_list = []
    for i in pixels_in_previous_frame:  #i should (x,y)
        # shifting based on an origin at the center of the frame and computing the unit vector
        x = i[0] - w
        y = h - i[1]
        v = np.array([x, y, -focal_distance])
        uvector = v / np.linalg.norm(v)

        # rotate the uvector
        new_vector = quaternion.as_rotation_matrix(R).dot(uvector.T)

        ux = new_vector[0]
        uy = new_vector[1]
        uz = new_vector[2]
        # calculate angles that the unit vector makes with z axis and with xz plane
        uxz = np.sqrt(ux * ux + uz * uz)
        theta = np.arcsin(ux / uxz)  # z is never zero, theta is the angle wrt to y
        phi = np.arcsin(uy)          # x is the angle wrt to z
        # compute x,y (z = -focal length)
        xval = focal_distance*np.tan(theta)
        yval = focal_distance*np.tan(phi)
        # convert to top left origin
        xn = xval + w
        yn = h - yval
        if xn <= width and yn <= height:    # This is not correct; see saliency.py for the corrected funtion
            pos = (xn,yn)
            combo = (i,pos)
            new_list.append(combo)
        else:
            print(f"point {i}in old frame is outside the new frame at {xn},{yn}")
        return new_list


class agent_oreo(object):
    # constructor
    def __init__(self, scene, result_folder, depth_camera=False, loc_depth_cam = 'c', foveation=False):


        self.agent_config = habitat_sim.AgentConfiguration()
        # Left sensor - # oreo perspective - staring at -ive z
        self.left_sensor = habitat_sim.SensorSpec()
        self.left_sensor.sensor_type = habitat_sim.SensorType.COLOR
        self.left_sensor.resolution = sensor_resolution
        self.left_sensor.uuid = "left_rgb_sensor"
        self.left_sensor.position = [-eye_separation / 2, 0.0, 0.0]
        self.left_sensor.orientation = np.array([0.0,0.0,0.0], dtype=float)
        self.left_sensor_hfov = math.radians(int(self.left_sensor.parameters['hfov']))
        self.focal_distance = sensor_resolution[0]/(2*math.tan(self.left_sensor_hfov/2))

        # Right sensor - # oreo perspective - staring at -ive z
        self.right_sensor = habitat_sim.SensorSpec()
        self.right_sensor.sensor_type = habitat_sim.SensorType.COLOR
        self.right_sensor.resolution = sensor_resolution
        self.right_sensor.uuid = "right_rgb_sensor"
        self.right_sensor.position = [eye_separation / 2, 0.0, 0.0]
        self.right_sensor.orientation = np.array([0.0, 0.0, 0.0], dtype=float)
        self.right_sensor_hfov = math.radians(int(self.right_sensor.parameters['hfov']))
        if self.right_sensor_hfov != self.left_sensor_hfov:
            print("Warning - Right and Left Sensor widths are not identical!")

        # Depth camera - At the origin of the reference coordinate axes (habitat frame)
        if depth_camera:
            self.num_sensors = 3
            self.depth_sensor = habitat_sim.SensorSpec()
            self.depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
            self.depth_sensor.resolution = sensor_resolution
            self.depth_sensor.uuid = "depth_sensor"
            if loc_depth_cam == 'l':
                self.depth_sensor.position = self.left_sensor.position
            elif loc_depth_cam == 'r':
                self.depth_sensor.position = self.right_sensor.position
            else:
                self.depth_sensor.position = [0.0,0.0,0.0]

            self.depth_sensor.orientation = np.array([0.0, 0.0, 0.0], dtype=float)
            self.agent_config.sensor_specifications = [self.right_sensor, self.left_sensor, self.depth_sensor]
        else:
            self.num_sensors = 2
            self.agent_config.sensor_specifications = [self.right_sensor, self.left_sensor]

        self.backend_cfg = habitat_sim.SimulatorConfiguration()

        if foveation:
            self.backend_cfg.foveation_distortion = True

        #self.backend_cfg.scene.id = scene   #This works in older habitat versions
        self.backend_cfg.scene_id = scene #newer versions like the colab install

        self.destination = os.path.realpath(result_folder)
        if not os.path.isdir(self.destination):
            os.makedirs(self.destination)

        # Tie the backend of the simulator and the list of agent configurations (only one)
        self.sim_configuration = habitat_sim.Configuration(self.backend_cfg, [self.agent_config])
        self.sim = habitat_sim.Simulator(self.sim_configuration)
        self.agent_id = self.backend_cfg.default_agent_id
        self.agent = self.sim.get_agent(self.agent_id)
        self.initial_agent_state = self.agent.get_state()
        print(f"Agent rotation {self.initial_agent_state.rotation} Agent position {self.initial_agent_state.position}" )
        # agent_head_neck_rotation (not a part of habitat api to keep track of head/neck rotation wrt to the agent.
        # HabitatAI api agent rotation is not rotation of agent wrt to WCS followed by rotation of head/neck
        self.agent_head_neck_rotation = np.quaternion(1,0,0,0)

        #self.counter = 0  # counter for saccade file numbering
        # self.filename = self.create_unique_filename(scene)
        self.my_images = self.get_sensor_observations()
        self.start_image_filenameRGB = None
        self.start_image_filenameBGR = None
        return

    def reset_state(self):
        #Agent rotation quaternion(1, 0, 0, 0) - Agent position[0.9539339, 0.1917877, 12.163067]
        self.agent.set_state(self.initial_agent_state,infer_sensor_states=False)
        self.agent_head_neck_rotation = np.quaternion(1, 0, 0, 0)
        self.my_images = self.get_sensor_observations()


    def get_current_state(self):
        return self.agent.get_state()


    def get_current_state_with_head_neck_rotation(self):
        my_state = []
        my_state.append(self.get_agent_sensor_position_orientations())
        my_state.append(self.agent_head_neck_rotation)
        return my_state


    def setup_agent_state(self,new_state):
        new_agent_state = self.agent.get_state()
        new_agent_state.rotation = new_state[0]
        new_agent_state.position = new_state[1]
        new_agent_state.sensor_states["left_rgb_sensor"].rotation = new_state[2]
        new_agent_state.sensor_states["left_rgb_sensor"].position = new_state[1]
        new_agent_state.sensor_states["right_rgb_sensor"].rotation = new_state[3]
        new_agent_state.sensor_states["right_rgb_sensor"].position = new_state[1]
        return new_agent_state


    def restore_state(self,new_astate):
        self.agent.set_state(new_astate, infer_sensor_states=False)
        self.my_images = self.get_sensor_observations()


    def create_unique_filename(self, scene_file):
        ''':scene_file - The scene for habitat
        returns a unique filename constructed from timestamp, scene file name, initial agent position
        and intial agent orientation. This can be used in combination with the counter to create a
        numbered sequence of image files.'''

        date_string = datetime.now()
        file_postfix = str(date_string)[:19]       # filename will be different by seconds.
        file_postfix = file_postfix.replace(" ", "-")
        file_postfix = file_postfix.replace(":", "-")

        head, scene_name = os.path.split(scene_file)
        '''
        d = scene_name.find(".")
        if d != -1:
            scene_name = scene_name[0:d]
        '''
        '''
        initial_orn = quaternion.as_float_array(self.initial_agent_state.rotation)
        val1 = '_' + str(initial_orn[0])+ "-" + str(initial_orn[1]) + "-" + str(initial_orn[3]) \
               + "-" + str(initial_orn[3])
        initial_pos = self.initial_agent_state.position
        val2= str(initial_pos[0]) + "-" + str(initial_pos[1]) + "-" + str(initial_pos[2])
        my_file = file_prefix + "-" + scene_name + val1 + '_' + val2 + "--"
        '''
        my_file = dest_folder + "/" + scene_name + "^"+ file_postfix
        return my_file

    def get_agent_sensor_position_orientations(self):
        """
        :return:
        agent orientation = a quaternion
        agent position = numpy array
        num_sensors = 2 (left and right) or 3 (left, right, depth)
        sensor[num_sensor] orientation quaternions. The sensor_states are with respect to habitat frame WCS.
        Habitat frame is cameras staring at -ive z and the +ive y is UP.

        Internally the relative sensor orientation and translation with respect to agent are stored
        under _sensor but the sensor_states that are accessible are wrt habitat frame WCS.
        The sensor states (eyes) wrt to the agent state (head/neck) is NOT computed by this function.
        """

        agent_state = self.agent.get_state()
        agent_orientation = agent_state.rotation
        agent_position = agent_state.position
        s1 = agent_state.sensor_states["left_rgb_sensor"].rotation
        s2 = agent_state.sensor_states["right_rgb_sensor"].rotation

        if self.num_sensors == 3:
            s3 = agent_state.sensor_states["depth_sensor"].rotation
            return agent_orientation, agent_position, s1, s2, s3
        else:
            return agent_orientation, agent_position, s1, s2


    def rotate_head_neck(self, rot_quat, oreo_py):
        """
        rot_quat: rotation in quaternion of the head/neck with respect to it current orientation.
        oreo_py: The oreo pybullet simulation object. Use this function to rotate the head neck
        of the robot head. Since Habitatai does not distinguish between body and head_neck movement
        of the agent, it is tracked separately.
        There is limit on the range of head/neck movement whereas Agent body can rotate.
        To apply the limit, it needs to be tracked separately.
        """

        new_headneck_orn = self.agent_head_neck_rotation * rot_quat
        result = oreo_py.is_valid_head_neck_rotation(new_headneck_orn)
        if result == 1:
            current_agent_state = self.agent.get_state()
            # save the inverse of the agent rotation before the head-neck rotation
            inv_agent_rotation = current_agent_state.rotation.inverse()
            # Include head_neck rotation into agent rotation
            current_agent_state.rotation = current_agent_state.rotation*rot_quat
            # update the sensor states
            current_agent_state.sensor_states["left_rgb_sensor"].rotation = \
                current_agent_state.rotation*(
                            inv_agent_rotation*current_agent_state.sensor_states["left_rgb_sensor"].rotation)
            current_agent_state.sensor_states["right_rgb_sensor"].rotation = \
                current_agent_state.rotation*(
                            inv_agent_rotation*current_agent_state.sensor_states["right_rgb_sensor"].rotation)

            if self.num_sensors == 3:
                current_agent_state.sensor_states["depth_sensor"].rotation = \
                    current_agent_state.rotation*(
                                inv_agent_rotation*current_agent_state.sensor_states["depth_sensor"].rotation)

            self.agent.set_state(current_agent_state, infer_sensor_states=False)
            # keep track of the head_neck position after the rot.
            self.agent_head_neck_rotation = new_headneck_orn
            self.my_images = self.get_sensor_observations()
        else:
            print(f"Invalid head neck rotation. No rotation performed")

    def get_agent_rotation(self):
        ''' return the agent body rotation by removing the head/neck rotation from the agent rotation '''
        my_agent_state = self.agent.get_state()
        r = my_agent_state.rotation  # Agent rotation (body +head_neck) wrt Habitat frame
        inv_head_neck_rotation = self.rotate_head_neck.inverse()
        return r*inv_head_neck_rotation

    def rotate_sensors_wrt_to_current_sensor_pose_around_Y(self, direction='ccw', my_angle=np.pi / 20):
        """
        :param direction: 'cw' or 'ccw'
        :param my_angle: The angle by which to rotate sensors from its current position - y axis
        :return:
        """

        if direction == 'ccw':
            rot_quat = quaternion.from_rotation_vector([0.0, my_angle, 0.0])
        elif direction == 'cw':
            rot_quat = quaternion.from_rotation_vector([0.0, -my_angle, 0.0])
        else:
            return
        self.rotate_sensors_wrt_to_current_sensor_pose([rot_quat,rot_quat,rot_quat])
        return


    def rotate_sensors_wrt_to_current_sensor_pose(self, sensor_rotations):
        """
        :param sensor_rotations: quaternions to rotate sensors from its current position
        :return:
        """

        my_agent_state = self.agent.get_state()
        # sensor states = Rotation of Agent wrt to habitat * rotation sensor wrt to agent
        my_agent_state.sensor_states["left_rgb_sensor"].rotation = \
            my_agent_state.sensor_states["left_rgb_sensor"].rotation * sensor_rotations[0]
        my_agent_state.sensor_states["right_rgb_sensor"].rotation = \
            my_agent_state.sensor_states["right_rgb_sensor"].rotation * sensor_rotations[1]
        if self.num_sensors == 3:
            my_agent_state.sensor_states["depth_sensor"].rotation = \
                my_agent_state.sensor_states["depth_sensor"].rotation * sensor_rotations[2]

        self.agent.set_state(my_agent_state, infer_sensor_states=False)
        self.my_images = self.get_sensor_observations()
        return


    def rotate_sensor_absolute_wrt_agent(self, sensors_rotation):
        """
        The agent position or orientation is unchanged.
        :param sensors_rotation: list of quaternions that specify rotation of sensor camera wrt agent frame
        :return: nothing
        The sensor_states within agent states keep the rotation of sensors wrt habitat frame.
        The relative rotation of the sensor wrt to agent is under protected '_sensor'
        which is modified when infer_sensor_states=False
        """
        agent_state = self.agent.get_state()
        agent_orn = agent_state.rotation
        agent_state.sensor_states["left_rgb_sensor"].rotation = agent_orn * sensors_rotation[0]
        agent_state.sensor_states["right_rgb_sensor"].rotation = agent_orn * sensors_rotation[1]
        if self.num_sensors ==3:
            agent_state.sensor_states["depth_sensor"].rotation = agent_orn * sensors_rotation[2]

        self.agent.set_state(agent_state, infer_sensor_states=False)
        self.my_images = self.get_sensor_observations()
        return

    def rotate_sensor_wrt_habitat(self, sensors_rotation):
        """
        The agent position or orientation is unchanged.
        :param sensors_rotation: list of quaternions that specify rotation of sensor camera wrt habitat frame
        :return: nothing
        The sensor_states within agent states keep the rotation of sensors wrt habitat frame.
        The relative rotation of the sensor wrt to agent is under protected '_sensor'
        which is modified when infer_sensor_states=False
        :return:
        """

        my_agent_state = self.agent.get_state()
        my_agent_state.sensor_states["left_rgb_sensor"].rotation = sensors_rotation[0]
        my_agent_state.sensor_states["right_rgb_sensor"].rotation = sensors_rotation[1]
        my_agent_state.sensor_states["depth_sensor"].rotation = sensors_rotation[2]
        self.agent.set_state(my_agent_state, infer_sensor_states=False)
        self.my_images = self.get_sensor_observations()
        return


    def move_and_rotate_agent(self, rotation, move=[0.0, 0.0, 0.0], ref="relative"):
        """
        The sensors orientation or position with respect to agent frame is unchanged.
        :param rotation: quaternion
        :param move: list
        :param ref: relative - relative to current agent position/orientation or absolute wrt to habitat
        :return:
        """

        my_agent_state = self.agent.get_state()
        r1 = my_agent_state.rotation  # Agent rotation wrt Habitat frame
        t1 = my_agent_state.position  # Agent translation wrt Habitat frame
        h1 = homogenous_transform(quaternion.as_rotation_matrix(r1), t1.tolist())
        h1_inv = inverse_homogenous_transform(h1)
        h = homogenous_transform(quaternion.as_rotation_matrix(rotation), move)
        if ref == "relative":
            new_h1 = h1.dot(h)
        elif ref == "absolute":
            new_h1 = h

        new_r1 = quaternion.from_rotation_matrix(new_h1[0:3, 0:3])
        new_t1 = new_h1[0:3, 3].T
        my_agent_state.rotation = new_r1
        my_agent_state.position = new_t1

        corr_h = new_h1.dot(h1_inv)         # Matrix multiplication is associative

        h2_left = homogenous_transform(quaternion.as_rotation_matrix(
            my_agent_state.sensor_states["left_rgb_sensor"].rotation),(my_agent_state.sensor_states[
            "left_rgb_sensor"].position).tolist())
        s_left = corr_h.dot(h2_left)
        my_agent_state.sensor_states["left_rgb_sensor"].rotation = quaternion.from_rotation_matrix(s_left[0:3,0:3])
        my_agent_state.sensor_states["left_rgb_sensor"].position = s_left[0:3, 3].T

        h2_right = homogenous_transform(quaternion.as_rotation_matrix(
            my_agent_state.sensor_states["right_rgb_sensor"].rotation),(my_agent_state.sensor_states[
            "right_rgb_sensor"].position).tolist())
        s_right = corr_h.dot(h2_right)
        my_agent_state.sensor_states["right_rgb_sensor"].rotation = quaternion.from_rotation_matrix(s_right[0:3, 0:3])
        my_agent_state.sensor_states["right_rgb_sensor"].position = s_right[0:3, 3].T

        if self.num_sensors == 3:
            h2_depth = homogenous_transform(quaternion.as_rotation_matrix(
                my_agent_state.sensor_states["depth_sensor"].rotation), (my_agent_state.sensor_states[
                "depth_sensor"].position).tolist())
            s_depth = corr_h.dot(h2_depth)
            my_agent_state.sensor_states["depth_sensor"].rotation = quaternion.from_rotation_matrix(
                s_depth[0:3, 0:3])
            my_agent_state.sensor_states["depth_sensor"].position = s_depth[0:3, 3].T

        self.agent.set_state(my_agent_state, infer_sensor_states=False)
        self.my_images = self.get_sensor_observations()

    def compute_uvector_for_image_point(self, x_pos, y_pos):
        """
        The x, y, z values are expressed in pixels. The purpose is to compute unit vector.
        z_pos is given by the distance 'f' from the principle point to the sensor image.
        :param x_pos: x is the width position in pixels where 0 is at the top left.
        :param y_pos: y is the height position in pixels where 0 is at the top left
        :return: np array, a unit vector pointing in the direction of the image point
        """

        #shifting the origin to width/2, height/2
        #xval = x_pos - (self.left_sensor.resolution[0]/2)       # width
        #yval = (self.left_sensor.resolution[1]/2) - y_pos       # height

        xval, yval = self.to_orgin_at_frame_center(x_pos,y_pos)
        v = np.array([xval, yval, -self.focal_distance])
        unit_vec = v / np.linalg.norm(v)
        return unit_vec

    def to_orgin_at_frame_center(self, xt, yt):

        # shifting the origin to width/2, height/2
        xval = xt - (self.left_sensor.resolution[0] / 2)  # width
        yval = (self.left_sensor.resolution[1] / 2) - yt  # height
        return xval, yval


    def saccade_to_new_point(self, xLeft, yLeft, xRight, yRight, oreo_pyb_sim):
        """

        :param xLeft: x position in pixels of Left sensor frame with 0,0 at top left
        :param yLeft: y position in pixels of Left sensor frame with 0,0 at top left
        :param xRight: x position in pixels of Right Sensor frame with 0,0 at top left
        :param yRight: y position in pixels of Right Sensor frame with 0,0 at top left
        :param oreo_pyb_sim: An Oreo_Pybullet_Sim object that can confirm the eye head movements
        :return: 1 and rotates the sensor if it is within range or a 0
        """
        my_agent_state = self.agent.get_state()
        r1 = my_agent_state.rotation  # Agent rotation wrt Habitat frame
        t1 = my_agent_state.position  # Agent translation wrt Habitat frame
        h1 = homogenous_transform(quaternion.as_rotation_matrix(r1), t1.tolist())
        h1_inv = inverse_homogenous_transform(h1)
        r1_inv = quaternion.from_rotation_matrix(h1_inv[0:3,0:3])

        print(f"Agent Rotation {r1}")
        '''
        First, can the eyes saccade to the new position without changing the head/neck orientation.
        Obtain the yaw and pitch for the proposed direction for the Sensor wrt to Agent Frame.
        Determine the rotation wrt to current sensor orn. Then extract the orn wrt to agent.
        This gives the rotation from agent to new sensor position.
        Compute the uvector pointing in the new sensor direction wrt agent frame
        Get the yaw and pitch.
        '''

        # Left
        new_dir_left_sensorframe = self.compute_uvector_for_image_point(xLeft, yLeft)
        rotation_agent_to_leftsensor = \
            quaternion.as_rotation_matrix(r1_inv * my_agent_state.sensor_states["left_rgb_sensor"].rotation)
        #print(f"Left Sensor Rotation{my_agent_state.sensor_states['left_rgb_sensor'].rotation}")
        new_dir_leftsensor_wrt_agentframe = rotation_agent_to_leftsensor.dot(new_dir_left_sensorframe.T)
        new_dir_left_pyBframe = rotatation_matrix_from_Pybullet_to_Habitat().dot(new_dir_leftsensor_wrt_agentframe.T)
        yaw_lefteye_pyB, pitchlefteye_pyB = oreo.compute_yaw_pitch_from_vector(new_dir_left_pyBframe)
        print(f"Desire to move to yawL = {yaw_lefteye_pyB}, pitchL = {pitchlefteye_pyB}")

        # Right
        new_dir_right_sensorframe = self.compute_uvector_for_image_point(xRight, yRight)
        rotation_agent_to_rightsensor = \
            quaternion.as_rotation_matrix(r1_inv * my_agent_state.sensor_states["right_rgb_sensor"].rotation)
        #print(f"Right Sensor Rotation{my_agent_state.sensor_states['right_rgb_sensor'].rotation}")
        new_dir_rightsensor_wrt_agentframe = rotation_agent_to_rightsensor.dot(new_dir_right_sensorframe.T)
        new_dir_right_pyBframe = rotatation_matrix_from_Pybullet_to_Habitat().dot(new_dir_rightsensor_wrt_agentframe.T)
        yaw_righteye_pyB, pitchrighteye_pyB = oreo.compute_yaw_pitch_from_vector(new_dir_right_pyBframe )
        print(f"Desire to move to yawR = {yaw_righteye_pyB}, pitchR = {pitchrighteye_pyB}")
        result = oreo_pyb_sim.is_valid_saccade([yaw_lefteye_pyB, pitchlefteye_pyB,yaw_righteye_pyB, pitchrighteye_pyB])
        if result[0] == 1:
            print("Moving Sensors to new position")
            leftSensorRotation_wrt_agent = result[1]
            rightSensorRotation_wrt_agent = result[2]
            self.rotate_sensor_absolute_wrt_agent(
                    [leftSensorRotation_wrt_agent, rightSensorRotation_wrt_agent,leftSensorRotation_wrt_agent])
            my_agent_state = self.agent.get_state()
            AL = my_agent_state.sensor_states['left_rgb_sensor'].rotation
            AR = my_agent_state.sensor_states['right_rgb_sensor'].rotation
            #print(f"After Rotation Left Sensor Rotation{AL}")
            #print(f"After Rotation Right Sensor Rotation{AR}")
            v1 = np.array([0.0,0.0,-1.0])
            AL_N = quaternion.as_rotation_matrix(r1_inv*AL).dot(v1.T)
            AR_N = quaternion.as_rotation_matrix(r1_inv*AR).dot(v1.T)
            AL_NN = rotatation_matrix_from_Pybullet_to_Habitat().dot(AL_N.T)
            AR_NN = rotatation_matrix_from_Pybullet_to_Habitat().dot(AR_N.T)
            yl,pl = oreo.compute_yaw_pitch_from_vector(AL_NN)
            yr,pr = oreo.compute_yaw_pitch_from_vector(AR_NN)
            print(f"After rotation Left Eye yaw = {yl} pitch = {pl}")
            print(f"After rotation Right Eye yaw = {yr} pitch = {pr}")
            return 1
        elif result[0] == 0:
            print("Not Moving Sensors to new position")
            return 0
            # Add head neck movement to move the Agent frame to the new point



    def get_sensor_observations(self):
        """

        :return: dict with sensor id as key and it values as ndarray (sensor_resolution (512 x 512), 4) for rgb sensors
        and depth resolution.
        The output from sensor.get_observations() is RGB
        """
        for _, sensor in self.sim._sensors.items():
            sensor.draw_observation()

        observations = {}
        for sensor_uuid, sensor in self.sim._sensors.items():
            observations[sensor_uuid] = sensor.get_observation()
        rgb_left = observations["left_rgb_sensor"]
        rgb_right = observations["right_rgb_sensor"]

        if self.num_sensors == 3:
            depth = observations["depth_sensor"]
            return rgb_left, rgb_right, depth
        else:
            return rgb_left, rgb_right


    def save_both_views(self,image_filename):

        output = []
        a = self.get_agent_sensor_position_orientations()
        output.append(a[0])  # Agent orn
        output.append(a[1])  # Agent position
        output.append(self.agent_head_neck_rotation)
        output.append(a[2])  # lefteye orientation
        output.append(a[3])  # righteye orientation
        # sensor res. hfov, focal distance - same for left, right and depth
        output.append(self.left_sensor.resolution)
        output.append(self.left_sensor_hfov)
        output.append(self.focal_distance)
        images = self.my_images[0][..., 0:3], self.my_images[1][..., 0:3]
        output.append(images)           # Left and Right RGB images

        self.start_image_filenameRGB = image_filename + "RGB"
        self.start_image_filenameBGR = image_filename + "BGR"
        try:
            with open(self.start_image_filenameRGB, "wb") as f:
                pickle.dump(output, f)
                print(f"Saved Image file {self.start_image_filenameRGB}")
        except IOError as e:
            print(f"Failure: To open/write image and data file {self.start_image_filenameRGB}")
            return 0

        del output[-1]
        images = self.my_images[0][..., 0:3][..., ::-1], self.my_images[1][..., 0:3][..., ::-1]
        output.append(images)
        try:
            with open(self.start_image_filenameBGR, "wb") as f:
                pickle.dump(output, f)
                print(f"Saved Image file {self.start_image_filenameBGR}")
        except IOError as e:
            print(f"Failure: To open/write image and data file {self.start_image_filenameBGR}")
            return 0
        return 1

    def capture_start_image_for_saliency(self):
        self.my_images = self.get_sensor_observations()
        new_file = self.create_unique_filename(self.backend_cfg.scene_id)
        self.save_both_views(new_file)


    def capture_fixation_image(self, processed_salfile, img_num):
        salpoint_data = saliency.get_salpoints(processed_salfile)
        # salpoint_data is a list  = [agent orientation, agent Position, robot_head_neck_rotation,
        # left_image, lefteye Rotation, list of x,y points, right_image, righteye Rotation, list of x,y points]
        if salpoint_data is None:
            return
        else:
            start_image_agent_state = self.setup_agent_state \
                ([salpoint_data[0], salpoint_data[1], salpoint_data[4], salpoint_data[7]])
            self.restore_state(start_image_agent_state)
            self.my_images = self.get_sensor_observations()
            i = salpoint_data[5][img_num]
            new_x = i[1]        # column value is x or width
            new_y = i[0]        # row value is y or height
            success = self.saccade_to_new_point(new_x, new_y, new_x, new_y, pybullet_sim)
            if success == 1:
                fig, ax = plt.subplots()
                ax.imshow(self.my_images[0])
                plt.show()
                aorn, apos, l_sensor_orn, _ = self.get_agent_sensor_position_orientations()
                val1 = [aorn, apos, oreo_in_habitat.agent_head_neck_rotation, l_sensor_orn,
                        self.my_images[0]]
            else:
                return
            i = salpoint_data[8][img_num]
            self.restore_state(start_image_agent_state)
            new_x = i[1]
            new_y = i[0]
            success = self.saccade_to_new_point(new_x, new_y, new_x, new_y, pybullet_sim)
            if success == 1:
                aorn, apos, _, r_sensor_orn = self.get_agent_sensor_position_orientations()
            return

    def capture_images_for_fixations(self, processed_salfile):

        robot_current_state = self.get_current_state()      # saving the current robot state
        # compare processed_salfile and scene to make sure that it corresponds to the right initial image scene
        _, scene_name = os.path.split(self.backend_cfg.scene_id)
        _, scenename_salfile = os.path.split(processed_salfile)
        d = scenename_salfile.find("^")
        if d == -1:
            print(f"The saliency file name {scenename_salfile} is missing the ^ char")
            print(f"Not capturing images from salient saccade points")
            return None
        else:
            if scene_name != scenename_salfile[0:d]:
                print(f"Saliency file {scenename_salfile} does not belong to scene {scene_name}")
                return None
            else:
                salpoint_data = saliency.get_salpoints(processed_salfile)
                # salpoint_data is a list  = [agent orientation, agent Position, robot_head_neck_rotation,
                # left_image, lefteye Rotation, list of x,y points, right_image, righteye Rotation, list of x,y points]
                if salpoint_data is None:
                    return None
                else:
                    start_image_agent_state = self.setup_agent_state\
                        ([salpoint_data[0],salpoint_data[1], salpoint_data[4], salpoint_data[7]])
                    # cycle through salient points in salpoint_data for all points for both images and generate
                    image_list_left = []
                    for i in salpoint_data[5]:  # salient point from Left eye image salient points
                        oreo_in_habitat.restore_state(start_image_agent_state)
                        new_x = i[1]    # x-axis is column (width) and y-axis is row (height) of the image
                        new_y = i[0]
                        success = oreo_in_habitat.saccade_to_new_point(new_x, new_y, new_x, new_y, pybullet_sim)
                        if success == 1:
                            aorn, apos, l_sensor_orn, _ = oreo_in_habitat.get_agent_sensor_position_orientations()
                            val = [aorn, apos, oreo_in_habitat.agent_head_neck_rotation, l_sensor_orn,
                                   oreo_in_habitat.my_images[0]]
                            image_list_left.append([i, val])
                        else:
                            image_list_left.append([i, None])

                    image_list_right = []
                    for i in salpoint_data[8]:  # right eye image
                        oreo_in_habitat.restore_state(start_image_agent_state)
                        new_x = i[1]
                        new_y = i[0]
                        success = oreo_in_habitat.saccade_to_new_point(new_x, new_y, new_x, new_y, pybullet_sim)
                        if success == 1:
                            aorn, apos, _, r_sensor_orn = oreo_in_habitat.get_agent_sensor_position_orientations()
                            val = [aorn, apos, oreo_in_habitat.agent_head_neck_rotation, r_sensor_orn,
                                   oreo_in_habitat.my_images[1]]
                            image_list_right.append([i, val])
                        else:
                            image_list_right.append([i, None])

                    output = [image_list_left,image_list_right]
                    image_filename = processed_salfile + "-images"
                    self.restore_state(robot_current_state)
                    try:
                        with open(image_filename, "wb") as f:
                            pickle.dump(output, f)
                            return output
                    except IOError as e:
                        print(f"Failure: To open/write image and data file {image_filename}")
                        return None

"""
    def get_start_image_filename(self):
        return self.start_image_filename
"""

class OreoPyBulletSim(object):
    def __init__(self, sim_path = "./"):
        self.oreo = oreo.Oreo_Robot(True, False, sim_path, "assembly.urdf", False)
        self.oreo.InitModel()
        self.oreo.InitManCtrl()
        # self.oreo.get_max_min_yaw_pitch_values()
        # read the interpolator function from pickle file
        b = self.oreo.read_interpolator_functions()
        if b == 0:
            a = self.oreo.read_oreo_yaw_pitch_actuator_data()
            if a == 0:
                print("Building scan data takes minutes ....")
                self.oreo.build_oreo_scan_yaw_pitch_actuator_data()
            self.oreo.produce_interpolators()
        return

    def is_valid_saccade(self, angles):
        """
        :param angles: left yaw, pitch, right yaw, pitch angles as a tuple wrt to agent frame
        :return: a tuple 1 (success), left, right rotations or 0 (failure) where 001 sequence
        represents collision while 000 is out of range
        :param angles:
        :return:
        """
        val = self.oreo.get_actuator_positions_for_a_given_yaw_pitch(angles)
        if val[0] == 1:
            accuator_positions = val[1:]
            num_collision, lefteye_orn, righteye_orn = self.oreo.move_eyes_to_position_and_return_orn(accuator_positions)
            if num_collision == 0:
                leftSensor_wrt_Agent = compute_eye_saccade_from_PyBframe(lefteye_orn)
                rightSensor_wrt_Agent = compute_eye_saccade_from_PyBframe(righteye_orn)
                return 1, leftSensor_wrt_Agent, rightSensor_wrt_Agent
            else:   # There is collision in moving to the point.
                print("Collision while trying to move actuators")
                return 0,0,1

        else:   # The given yaw pitch is outside the range of actuator values
            print("Angles(s) out of range")
            return 0,0,0      #out of range


    def is_valid_head_neck_rotation(self, rotation_quat):
        """
        rotation_quat: in quaternion, the orientation of head_neck
        Converting the Quaternion to Rotation matrix and taking the unit vector corresponding to z-axis
        and rotate move it to pyB frame to compute yaw and pitch.
        Yaw + to - 90 degrees, pitch is -55 to +70 translates to a range of 35 to 160 since pitch 0 is
        aligned to +z and 180 to -z of the PyBframe. FYI - Roll of +40 t0 -40 roll is not programmed in.
        """

        head_neck_orn_matrix = quaternion.as_rotation_matrix(rotation_quat)
        z_axis = -head_neck_orn_matrix[:, 2]
        view_direction = rotatation_matrix_from_Pybullet_to_Habitat().dot(z_axis.T)
        yaw_headneck_pyB, pitch_headneck_pyB = oreo.compute_yaw_pitch_from_vector(view_direction)
        if -90 < yaw_headneck_pyB < 90 and 35 < pitch_headneck_pyB < 160:
            return 1
        else:
            print(f"Out of range either yaw {yaw_headneck_pyB} or pitch {pitch_headneck_pyB}")
            return 0


if __name__ == "__main__":

    pybullet_sim = OreoPyBulletSim("/Users/rajan/mytest/")
    oreo_in_habitat = agent_oreo(scene, dest_folder, depth_camera=False, loc_depth_cam = 'c', foveation=False)
    delta_move = 0.1
    ang_quat = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
    delta_ang_ccw  = quaternion.from_rotation_vector([0.0, 2*np.pi/30,0.0])
    delta_ang_cw = quaternion.from_rotation_vector([0.0, -2*np.pi/30, 0.0])
    w = sensor_resolution[0]
    h = sensor_resolution[1]
    left = 1
    right = 1
    image_number = 0
    while (1):
        display_image(oreo_in_habitat.my_images)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        elif k == ord("0"):
            oreo_in_habitat.capture_start_image_for_saliency()
            continue
        elif k == ord("1"):
            #take the processed saliency file to capture salient images and related information
            processed_dir = "/Users/rajan/PycharmProjects/saliency/saliency_map/results/"
            for root, dirs, files in os.walk(processed_dir):
                for filename in files:
                    if "-sal-processed" in filename:
                        p_salfile = processed_dir + filename
                        related_images = oreo_in_habitat.capture_images_for_fixations(p_salfile)
            continue
        elif k == ord("2"):
            #take the processed saliency file to capture salient images and related information
            processed_dir = "/Users/rajan/PycharmProjects/saliency/saliency_map/results/"
            filename ="van-gogh-room.glb^2021-07-04-14-25-14-sal-processed"
            p_salfile = processed_dir + filename
            robot_current_state = oreo_in_habitat.get_current_state()  # saving the current robot state
            # compare processed_salfile and scene to make sure that it corresponds to the right initial image scene
            _, scene_name = os.path.split(oreo_in_habitat.backend_cfg.scene_id)
            _, scenename_salfile = os.path.split(p_salfile)
            d = scenename_salfile.find("^")
            if d == -1:
                print(f"The saliency file name {scenename_salfile} is missing the ^ char")
                print(f"Not capturing image from Fixation points")
            else:
                if scene_name != scenename_salfile[0:d]:
                    print(f"Saliency file {scenename_salfile} does not belong to scene {scene_name}")
                else:
                    img_data = oreo_in_habitat.capture_images_for_fixations(p_salfile)
                    oreo_in_habitat.restore_state(robot_current_state)
                    continue
        elif k == ord("3"):
            #take the processed saliency file to capture salient images and related information
            processed_dir = "/Users/rajan/PycharmProjects/saliency/saliency_map/results/"
            filename ="van-gogh-room.glb^2021-07-04-14-23-07-sal-processed"
            p_salfile = processed_dir + filename
            robot_current_state = oreo_in_habitat.get_current_state()  # saving the current robot state
            # compare processed_salfile and scene to make sure that it corresponds to the right initial image scene
            _, scene_name = os.path.split(oreo_in_habitat.backend_cfg.scene_id)
            _, scenename_salfile = os.path.split(p_salfile)
            d = scenename_salfile.find("^")
            if d == -1:
                print(f"The saliency file name {scenename_salfile} is missing the ^ char")
                print(f"Not capturing image from Fixation points")
            else:
                if scene_name != scenename_salfile[0:d]:
                    print(f"Saliency file {scenename_salfile} does not belong to scene {scene_name}")
                else:
                    if (image_number < 10):
                        limage = oreo_in_habitat.capture_fixation_image(p_salfile, image_number)
                        image_number += 1
                    continue
        elif k == ord('4'):
            oreo_in_habitat.restore_state(robot_current_state)
            continue
        elif k == ord('n'):
            oreo_in_habitat.reset_state()
            continue
        elif k == ord('f'):
            oreo_in_habitat.move_and_rotate_agent(ang_quat, [0.0, 0.0, -delta_move])
            continue
        elif k == ord('b'):
            oreo_in_habitat.move_and_rotate_agent(ang_quat, [0.0, 0.0, delta_move])
            continue
        elif k == ord('u'):
            oreo_in_habitat.move_and_rotate_agent(ang_quat, [0.0, delta_move, 0.0])
            continue
        elif k == ord('v'):
            oreo_in_habitat.move_and_rotate_agent(ang_quat, [0.0, -delta_move, 0.0])
            continue
        elif k == ord('s'):
            oreo_in_habitat.move_and_rotate_agent(ang_quat, [delta_move, 0.0, 0.0])
            continue
        elif k == ord('t'):
            oreo_in_habitat.move_and_rotate_agent(ang_quat, [-delta_move, 0.0, 0.0])
            continue
        elif k == ord('j'):
            # default agent position is 0.9539339  0.1917877 12.163067
            m = [0.9539339, 0.1917877, 11.0]
            oreo_in_habitat.move_and_rotate_agent(ang_quat, m, "absolute")
            continue

        elif k == ord('a'):
            oreo_in_habitat.move_and_rotate_agent(delta_ang_cw, [0.0,0.0,0.0])
            continue
        elif k == ord('c'):
            oreo_in_habitat.move_and_rotate_agent(delta_ang_ccw, [0.0, 0.0, 0.0])
            continue
        elif k == ord('8'):
            #oreo_in_habitat.reset_state()
            dc = oreo_in_habitat.compute_uvector_for_image_point(w/4, h/4)
            rot_quat = calculate_rotation_to_new_direction(dc)
            oreo_in_habitat.rotate_sensors_wrt_to_current_sensor_pose([rot_quat,rot_quat,rot_quat])
            continue
        elif k == ord('9'):
            #oreo_in_habitat.reset_state()
            dc = oreo_in_habitat.compute_uvector_for_image_point(3*w/4, 3*h/4)
            rot_quat = calculate_rotation_to_new_direction(dc)
            oreo_in_habitat.rotate_sensors_wrt_to_current_sensor_pose([rot_quat,rot_quat,rot_quat])
            continue
        elif k == ord('e'):
            oreo_in_habitat.reset_state()
            dc = oreo_in_habitat.compute_uvector_for_image_point(0, h/2)
            rot_quat = calculate_rotation_to_new_direction(dc)
            a = quaternion.from_rotation_vector([0, -1.0 * np.pi / 3, 0.0])
            next_quat = rot_quat * a
            print(f"Quaternion {rot_quat} at 0,h/2, next quat {next_quat}, a ={a}")
            oreo_in_habitat.rotate_sensors_wrt_to_current_sensor_pose([rot_quat, rot_quat, rot_quat])
            continue
        elif k == ord('w'):
            oreo_in_habitat.reset_state()
            dc = oreo_in_habitat.compute_uvector_for_image_point(w, h/2)
            rot_quat = calculate_rotation_to_new_direction(dc)
            print(f"Quaternion {rot_quat} at w,h/2")
            oreo_in_habitat.rotate_sensors_wrt_to_current_sensor_pose([rot_quat, rot_quat, rot_quat])
            continue
        elif k == ord('z'):
            oreo_in_habitat.saccade_to_new_point((w/2)+8,(h/2)+8,w/2,h/2, pybullet_sim)
            continue
        elif k == ord('p'):
            oreo_in_habitat.saccade_to_new_point((w / 2) - 8, (h / 2) - 8, w / 2, h / 2, pybullet_sim)
            continue
        elif k == ord('l'):
            print(f"Move Left count --{left}--")
            xL = (w/2)-8
            yL = h/2
            oreo_in_habitat.saccade_to_new_point(xL, yL, xL, yL, pybullet_sim)
            left += 1
            continue
        elif k == ord('r'):
            print(f"Move Right count --{right}-")
            right  += 1
            oreo_in_habitat.saccade_to_new_point((w/2)+8,h/2,(w/2)+8,h/2, pybullet_sim)
            continue
        elif k == ord('d'):
            oreo_in_habitat.saccade_to_new_point(w/2,(h/2)+8, w/2, (h/2)+8, pybullet_sim)
            continue
        elif k == ord('y'):
            oreo_in_habitat.rotate_head_neck(quaternion.from_rotation_vector([0,5*np.pi/180,0]),
                                             pybullet_sim)
            continue
        elif k == ord('x'):
            oreo_in_habitat.rotate_head_neck(quaternion.from_rotation_vector([0, -5*np.pi/180, 0]),
                                             pybullet_sim)
            continue
        elif k == ord('g'):
            oreo_in_habitat.rotate_head_neck(quaternion.from_rotation_vector([5*np.pi/180,0,0]),
                                             pybullet_sim)
            continue
        elif k == ord('h'):
            oreo_in_habitat.rotate_head_neck(quaternion.from_rotation_vector([-5*np.pi/180,0,0]),
                                             pybullet_sim)
            continue
        else:
            pass


