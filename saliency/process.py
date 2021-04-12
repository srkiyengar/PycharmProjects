import pickle
import os
import saliency
import matplotlib.pyplot as plt

fname = "./saliency_map/skokloster-castle--0.saliency"
my_scene_info = "./saliency_map/skokloster-castle--0"

my_block = 16
eye_hov = 120
total_points = 10


class process_image(object):

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
        self.R = scene_data[3]
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


if __name__ == "__main__":

    sal_object = saliency.saliency(my_scene_info, fname, my_block, eye_hov, total_points)

    my_sal_process = process_image(my_scene_info)
    img = my_sal_process.image
    my_rot = my_sal_process.R
    my_sal_process.save_salmap(sal_object.salmap)
    try:
        with open(my_sal_process.salmap_filename, "rb") as f:
            saved_salmap = pickle.load(f)
    except IOError as e:
        print("Failure: Loading pickle file {}".format(saved_salmap))

    fig, axes = plt.subplots(1, ncols=2)

    axes[0].imshow(img)
    axes[1].imshow(saved_salmap)
    plt.show()
    a=5