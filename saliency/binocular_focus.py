import matplotlib.pyplot as plt
import pickle
import numpy as np
from skimage.feature import match_template
import agent_oreo

datafile_location = "/Users/rajan/PycharmProjects/saliency/saliency_data/binocular_fixation/"
datafile_name = "van-gogh-room.glb^2022-02-20-03-29-18RGB0-sal-processed"

def read_file(myfilename):
    try:
        with open(myfilename, "rb") as f:
            data = pickle.load(f)
            return data
    except IOError as e:
        print("Failure: Loading pickle file {}".format(myfilename))
        return None

def write_file(myfilename, info):
    try:
        with open(myfilename, "wb") as f:
            pickle.dump(info, f)
            return myfilename
    except:
        print(f"Failure: To save saliency file {myfilename}")
        return None


def display_image_with_salpoints(image, sal_points, wide=6):

    '''
    :param image is the left eye
    :param sal_points list of 0 to 9(10 fixation points)
    '''

    val = 30
    img =np.copy(image)
    if img is not None:
        for i, p in enumerate(sal_points):
            if p is not None:
                r = p[2]
                c = p[3]
                img[r - wide:r + wide + 1, c - wide:c + wide + 1, 0] = 255  # RGB layers in the image
                img[r - wide:r + wide + 1, c - wide:c + wide + 1, 1] = i*val
                img[r - wide:r + wide + 1, c - wide:c + wide + 1, 2] = 255
            pass
    return img

def display_image_showing_match(l_image, fixation, r_image, location, wide=6):

    '''
    :param image is right eye
    :param location where the match was found with the left eye
    '''

    img_right = np.copy(r_image)
    r = location[0]
    c = location[1]
    img_right[r - wide:r + wide + 1, c - wide:c + wide + 1, 0] = 255  # RGB layers in the image
    img_right[r - wide:r + wide + 1, c - wide:c + wide + 1, 1] = 0
    img_right[r - wide:r + wide + 1, c - wide:c + wide + 1, 2] = 255

    img_left = np.copy(l_image)
    r = fixation[0]
    c = fixation[1]
    img_left[r - wide:r + wide + 1, c - wide:c + wide + 1, 0] = 255  # RGB layers in the image
    img_left[r - wide:r + wide + 1, c - wide:c + wide + 1, 1] = 0
    img_left[r - wide:r + wide + 1, c - wide:c + wide + 1, 2] = 255


    return np.concatenate((img_left, img_right),axis=1)

def get_image_and_salmap(filename):
    '''
    :param sal_processed_dir: which contains sal-processed pickled files each of which contain
    0 = [imageL, imageR]
    1 = [salmapL, salmapR]
    2 = [reduced_salmapL, reduced_salmapR]
    3 = [recreated_salmapL, recreated_salmapR]
    4 = [center_pointsL, center_pointsR]
    5 = focal_distance
    6 = agent_orn
    7 = agent_pos
    8 = [lefteye rotation, righteye rotation]
    9 = robot_head_neck_rotation
    :type sal_processed_dir: directory which should contain 11 sal-processed pickled file
    :return: a tuple consisting of an image list, a salmap list
    :rtype: tuple
    '''

    my_data = read_file(filename)
    my_agent_state = []
    if my_data is not None:
        imageL = my_data[0][0]
        imageR = my_data[0][1]
        pointsL = my_data[4][0]
        pointsR = my_data[4][1]
        my_agent_state.append(my_data[6])
        my_agent_state.append(my_data[7])
        my_agent_state.append(my_data[8][0])
        my_agent_state.append(my_data[8][1])

    return imageL, imageR, pointsL, pointsR, my_agent_state


if __name__ == "__main__":

    oreo_in_habitat = agent_oreo.agent_oreo(agent_oreo.scene, agent_oreo.dest_folder, agent_oreo.pyBfolder,
                                            depth_camera=False, loc_depth_cam='c', foveation=False)
    my_file = datafile_location + datafile_name
    l_image, r_image, l_points, r_points, agent_state = get_image_and_salmap(my_file)
    l_image_with_points = display_image_with_salpoints(l_image,l_points)
    r_image_with_points = display_image_with_salpoints(r_image, r_points)

    nrow = 1
    ncol = 2
    fig, ax = plt.subplots(nrow, ncol)
    ax[0].imshow(l_image_with_points)
    ax[1].imshow(r_image_with_points)
    plt.show()


    win_span = 5    # window span is 2*win_span + 1
    height = l_image.shape[0]
    width = l_image.shape[1]

    for i in l_points:
        fix_row = i[2]
        fix_col = i[3]
        print(f"Salient point {fix_row}, {fix_col}")
        r1 = fix_row - win_span
        r2 = fix_row + win_span
        c1 = fix_col - win_span
        c2 = fix_col + win_span
        if (0 <= r1 < height and 0 <= r2 < height and 0 <= c1 < width and 0 <= c2 < width):
            patch = l_image[r1:r2+1, c1:c2+1, :]
            plt.imshow(patch)
            plt.show()
            result = match_template(r_image, patch)
            loc = np.unravel_index(np.argmax(result), result.shape)
            print(f"Match Point {loc[0]}, {loc[1]}")
            match_image = display_image_showing_match(l_image,[fix_row,fix_col],r_image, loc, wide=6)
            plt.imshow(match_image)
            plt.show()
            new_image = oreo_in_habitat.capture_binocular_fixation_images(agent_state, fix_row, fix_col, loc[0], loc[1])
            if new_image is not None:
                plt.imshow(new_image)
                plt.show()
        else:
            print(f"Patch boundaries{r1}:{r2} or {c1}:{c2} should be within the image shape {l_image.shape}")



    a = 5