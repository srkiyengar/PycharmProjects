import pickle
import numpy as np
import skimage.measure
import math
import matplotlib.pyplot as plt
import os


fname = "./saliency_map/apartment_1--0.saliency"
my_scene_info = "./saliency_map/apartment_1--0"
'''
fname = "./saliency_map/van-gogh-room--0.saliency"
my_scene_info = "./saliency_map/van-gogh-room--0"
fname = "./saliency_map/apartment_1--0.saliency"
my_scene_info = "./saliency_map/apartment_1--0"
'''

my_block = 16
eye_hov = 120
total_points = 10
new_max = 150

def scale_image(some_image):
    vmin = some_image.min()
    vmax = some_image.max()
    some_image = (some_image - vmin) * new_max / (vmax - vmin)
    return some_image

def find_max_and_index(array_2d):
    #result = np.where(array_2d == np.amax(array_2d))
    result = np.nonzero(array_2d == np.amax(array_2d))
    r = result[0][0]
    c = result[1][0]
    return np.amax(array_2d), r, c


class saliency(object):

    def __init__(self, scene_info_file, salmap_file, block_dim, camera_hov, num_points):
        self.block_dim = block_dim
        try:
            with open(salmap_file, "rb") as f:
                self.salmap = pickle.load(f)
                try:
                    with open(scene_info_file, "rb") as f:
                        scene_data = pickle.load(f)
                except IOError as e:
                    print("Failure: Loading pickle file {}".format(scene_info_file))
                    exit(1)
        except IOError as e:
            print("Failure: Opening saliency file {}".format(fname))
            try:
                with open(scene_info_file, "rb") as f:
                    scene_data = pickle.load(f)
            except IOError as e:
                print("Failure: Loading pickle file {}".format(scene_info_file))
            finally:
                exit(1)

        self.salmap_file = salmap_file
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
        8 - images [0 = left, 1 = right, 2 = depth optionally set]
        '''
        self.scene = scene_data[8][0][:, :, ::-1]
        self.reduced_salmap = skimage.measure.block_reduce(self.salmap, block_size=(block_dim, block_dim), \
                                                           func=np.amax)
        self.reduced_salmap = scale_image(self.reduced_salmap)
        self.reduced_sal_points = []
        # Macular angle is 18 out of HFOV (120); d should be 5
        # We ignore (d-1)/2 pixels on horizontal and vertical directions of a salient pixel
        # When dim = 16x16 for map reduction, d=5 means 2 pixels on each side, 32 pixels in 512 scale
        # Each red. image pixel = 16 regular approximately 3.5 degrees
        self.d = math.ceil((18.0 * self.salmap.shape[0]) / (camera_hov * block_dim))
        #print(f"The dimension d = {self.d}")
        # compute salient points list
        for i in range(num_points):
            val, r, c = find_max_and_index(self.reduced_salmap)
            self.reduced_sal_points.append((val, r, c))
            self.zero_around_macular_center(r, c, i)
        self.num_points = num_points
        for i in range(num_points):
            self.reduced_salmap[self.reduced_sal_points[i][1], self.reduced_sal_points[i][2]] = 255-i*10
        
        self.recreated_salmap = scale_image(np.copy(self.salmap))
        self.center_points = []
        self.show_sal_point_for_full_image()
        self.output_filename = "empty"


    def zero_around_macular_center(self, r, c, i):
        f = int((self.d - 1) / 2)
        self.reduced_salmap[r-f:r+f+1,c-f:c+f+1] = 0


    def show_sal_point_for_full_image(self):
        count = 0
        for scaled_val, r, c in self.reduced_sal_points:
            start_r = r * self.block_dim
            start_c = c * self.block_dim
            '''
            recreated_image[start_r:start_r+block_size,start_c:start_c+block_size] = \
                recreated_image[start_r:start_r + block_size, start_c:start_c + block_size]/2
            '''
            val = 255 - 10 * count
            self.recreated_salmap[start_r:start_r + self.block_dim, start_c:start_c + self.block_dim] = val
            true_val = self.salmap[start_r + 16, start_c + 16]
            self.center_points.append((true_val, scaled_val, start_r + 16, start_c + 16))
            count += 1

        return


    def save_output(self):
        head, tail = os.path.split(self.salmap_file)
        d = tail.find(".")
        if d != -1:
            tail = tail[0:d]

        output_filename = head + '/' + tail + "-salicency-info"
        output = []
        output.append(self.recreated_salmap)
        output.append(self.center_points)
        try:
            with open(output_filename, "wb") as f:
                pickle.dump(output, f)
                self.output_filename = output_filename

        except IOError as e:
            print(f"Failure: To open/write image and data file {output_filename}")


    def read_output(self):
        if self.output_filename != "empty":
            try:
                with open(self.output_filename, "rb") as f:
                    sal_info = pickle.load(f)
            except IOError as e:
                print(f"Failure: Loading pickle file {self.output_filename}")
                exit(1)
        else:
            print(f"The saliency info output file has not been saved. Run save_output()")

        return sal_info

    def get_output_filename(self):
        return self.output_filename

if __name__ == "__main__":

    sal_object = saliency(my_scene_info, fname, my_block, eye_hov, total_points)
    fig = plt.figure(figsize=(8, 8))
    r1c1 = fig.add_subplot(2, 2, 1)
    r1c2 = fig.add_subplot(2, 2, 2)
    r2c1 = fig.add_subplot(2, 2, 3)
    r2c2 = fig.add_subplot(2, 2, 4)

    r1c1.imshow(sal_object.scene)
    r1c2.imshow(sal_object.salmap)
    r2c1.imshow(sal_object.reduced_salmap)
    r2c2.imshow(sal_object.recreated_salmap)
    plt.show()
    sal_object.save_output()
    s_info = sal_object.read_output()

    fig2, axes = plt.subplots(1, ncols=2)

    axes[0].imshow(sal_object.recreated_salmap)
    axes[1].imshow(s_info[0])
    plt.show()
    print(s_info[1])

