import pickle
import os
import saliency
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import math




my_block = 16
eye_hov = 120
total_points = 10


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

    R = R2.inverse() * R1  # Rotation from current frame to previous frame
    w = width / 2
    h = height / 2
    new_list = []
    for i in pixels_in_previous_frame:  # i should (x,y)
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
        theta = np.arcsin(ux / uxz)  # z is never zero, theta is the rotation angle about y-axis - yaw angle
        phi = np.arcsin(uy)  # x is the angle about x - pitch angle
        # compute x,y (z = -focal length)
        xval = focal_distance * np.tan(theta)
        yval = focal_distance * np.tan(phi)
        # convert to top left origin
        xn = math.floor(xval + w)
        yn = math.floor(h - yval)
        if xn <= width and yn <= height:    # logic is suspect;
            pos = (xn, yn)
            combo = (i, pos)
            new_list.append(combo)
        else:
            new_list.append(None)
            print(f"point {i}in old frame is outside the new frame at {xn},{yn}")
    return new_list

# So that we can perform only Deep Gaze II in collab
class process_image(object):
    ''':arg
    Since salmap comes from Deep Gaze II through collab, saving the salmap is not in the __init__.
    creating this object will allow us to run Deep Gaze II without running habitat-sim in collab.
    It can also be used to create a unified object which saves everything in "-processed". image, salmap, centerpoints,
    rotation information.
    '''

    def __init__(self,image_info_file):
        self.fname = image_info_file
        try:
            with open(image_info_file, "rb") as f:
                scene_data = pickle.load(f)
        except IOError as e:
            print("Failure: Loading pickle file {}".format(image_info_file))
            exit(1)

        '''    
        scene_data
            0 - Agent orn
            1 - Agent position
            2 - agent_head_neck_rotation
            3 - lefteye orientation
            4 - righteye orientation
            5 - left_sensor.resolution
            6 - left_sensor_hfov
            7 - focal_distance
            8 - images[0 = left, 1 = right, 2 = depth optionally set]
        '''
        self.image = scene_data[8][0][:, :, ::-1]
        self.lefteyeR = scene_data[3]
        self.hov = scene_data[6]
        self.focal_distance = scene_data[7]

        head, tail = os.path.split(image_info_file)
        d = tail.find(".")
        if d != -1:
            tail = tail[0:d]
        self.output_filename = head + '/' + tail + "-processed"
        self.salmap_filename = head + '/' + tail + ".saliency"



    def save_salmap(self,salval):
        '''
        :param salval: saliency map
        :type: ndarray
        :return:
        '''
        try:
            with open(self.salmap_filename, "wb") as f:
                pickle.dump(salval, f)
        except:
            print(f"Failure: To save saliency file {self.salmap_filename}")

    def save_all(self,salmap, recreated_map, centers):
        all = []
        all.append(self.image)
        all.append(salmap)
        all.append(recreated_map)
        all.append(centers)
        all.append(self.lefteyeR)
        all.append(self.focal_distance)

        try:
            with open(self.output_filename, "wb") as f:
                pickle.dump(all, f)
        except:
            print(f"Failure: To save saliency file {self.output_filename}")

class saliency_comparison(object):

    def __init__(self,ref_datafile):
        self.ref_data =self.read_info(ref_datafile)
        self.ref_datafile = ref_datafile
        self.current_data = []
        self.images = []
        self.salmap = []
        self.recon_image = []
        self.consolidated_file = []


    def load_current_datafile(self,current_datafile):
        self.current_data = self.read_info(current_datafile)
        print(f"The data from {current_datafile} is loaded")

    def read_info(self, datafile):
        try:
            with open(datafile, "rb") as f:
                scene_data = pickle.load(f)
        except IOError as e:
            print("Failure: Loading pickle file {}".format(datafile))
            exit(1)
        return scene_data

    def save_sal_comparsion(self, mydata):
        head, tail = os.path.split(self.ref_datafile)
        d = tail.find("-")
        if d != -1:
            tail = tail[0:d]
        sal_compare = head + '/' + tail + "-saliency-comparison"
        self.consolidated_file = sal_compare

        try:
            with open(sal_compare, "wb") as f:
                pickle.dump(mydata, f)
        except:
            print(f"Failure: To save saliency file {sal_compare}")

    def read_sal_comparison(self):
        try:
            with open(self.consolidated_file, "rb") as f:
                consolidated_data = pickle.load(f)
        except IOError as e:
            print("Failure: Loading pickle file {}".format(self.consolidated_file))
            exit(1)
        return consolidated_data


if __name__ == "__main__":

    num_files = 4
    fileR = "./saliency_map/apartment_1--0-processed"
    #fileC = "./saliency_map/skokloster-castle--5-processed"
    fileC1 = "./saliency_map/apartment_1--"
    fileC2 = "-processed"
    my_comparison = saliency_comparison(fileR)
    w, h, d = my_comparison.ref_data[0].shape
    sal_points = my_comparison.ref_data[3]
    # Appending images, salmap and reconstructed images
    my_comparison.images.append(my_comparison.ref_data[0])
    my_comparison.salmap.append(my_comparison.ref_data[1])
    my_comparison.recon_image.append(my_comparison.ref_data[2])
    sal_table = []
    ref_point_list = []
    for i in sal_points:
        point_data = []
        p = (i[2], i[3])      # xpos, ypos, saliency value
        ref_point_list.append(p)
        point_data.append(p)
        point_data.append(i[0])
        sal_table.append(point_data)
    ref_R = my_comparison.ref_data[4]
    focal_distance = my_comparison.ref_data[5]
    my_array = np.ndarray((10,num_files), dtype=float)
    for i in range(1,num_files,1):
        fileC = fileC1 + str(i) +fileC2
        my_comparison.load_current_datafile(fileC)
        my_comparison.images.append(my_comparison.current_data[0])
        my_comparison.salmap.append(my_comparison.current_data[1])
        my_comparison.recon_image.append(my_comparison.current_data[2])
        cur_R = my_comparison.current_data[4]
        new_points = compute_pixel_in_current_frame(ref_R, cur_R, ref_point_list, focal_distance, w, h)
        k = 0
        for j in new_points:
            xv = j[1][0]
            yv = j[1][1]
            if xv < w and yv < h:
                point_data = my_comparison.current_data[1][xv,yv]
                sal_table[k].append(point_data)
                k += 1
    x = np.linspace(1, 5, 4, endpoint=False)
    fig, ax = plt.subplots()
    for m in range(9):
        my_array[m] = sal_table[m][1:]
        my_label = "point {}, {}".format(sal_table[m][0][0], sal_table[m][0][1])
        ax.plot(x, my_array[m], label=my_label)
    plt.xticks([1,2,3,4])
    ax.legend()
    plt.xlabel('Saccades')
    plt.ylabel('Saliency')
    plt.title('Apartment 1')
    consolidated = []
    consolidated.append(my_comparison.images)
    consolidated.append(my_comparison.salmap)
    consolidated.append(my_comparison.recon_image)
    consolidated.append(ref_point_list)
    consolidated.append(my_array)
    my_comparison.save_sal_comparsion(consolidated)
    stored_consolidated = my_comparison.read_sal_comparison()
    a=5
    '''

    fname = "./saliency_map/van-gogh-room--4.saliency"
    my_scene_info = "./saliency_map/van-gogh-room--4"
    scene_sal_object = saliency.saliency(my_scene_info, fname, my_block, eye_hov, total_points)
    scene_sal_process = process_image(my_scene_info)
    #img = scene_sal_process.image
    #my_rot = scene_sal_process.R
    #scene_sal_process.save_salmap(sal_object.salmap)
    scene_sal_process.save_all(scene_sal_object.salmap, scene_sal_object.recreated_salmap, scene_sal_object.center_points)
    try:
        with open(scene_sal_process.output_filename, "rb") as f:
            scene_all_data = pickle.load(f)
    except IOError as e:
        print("Failure: Loading pickle file {}".format(scene_sal_process.output_filename))
    fig, axes = plt.subplots(1, ncols=3)
    axes[0].imshow(scene_all_data[0])
    axes[1].imshow(scene_all_data[1])
    axes[2].imshow(scene_all_data[2])
    plt.show()
    print(f"Image salient points {scene_all_data[3]}")
    print(f"Eye rotation {scene_all_data[4]}")
    print(f"Focal Distance is {scene_all_data[5]}")
    a = 5
    '''



