import matplotlib.pyplot as plt
import pickle
import numpy as np
import quaternion
from skimage.feature import match_template
from skimage.color import rgb2gray
import agent_oreo

datafile_location = "/Users/rajan/PycharmProjects/saliency/saliency_data/binocular_fixation/"
datafile_name = "van-gogh-room.glb^2022-02-20-03-29-18RGB0-sal-processed"

#plt.gca().imshow(result, alpha=0.2)
#m = plt.gca().matshow((log_density_prediction[0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
#plt.colorbar(m)

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

def display_image_showing_match(l_image, fixation, r_image, location, cyc_img, cyc_p, wide=4):

    '''
    :param image is right eye
    :param location where the match was found with the left eye
    '''

    img_right = np.copy(r_image)
    r = location[0]
    c = location[1]
    #print(f"right image pixel row:{r} col:{c}")
    img_right[r - wide:r + wide + 1, c - wide:c + wide + 1, 0] = 255  # RGB layers in the image
    img_right[r - wide:r + wide + 1, c - wide:c + wide + 1, 1] = 0
    img_right[r - wide:r + wide + 1, c - wide:c + wide + 1, 2] = 255

    img_left = np.copy(l_image)
    r = fixation[0]
    c = fixation[1]
    #print(f"left image pixel row:{r} col:{c}")
    img_left[r - wide:r + wide + 1, c - wide:c + wide + 1, 0] = 255  # RGB layers in the image
    img_left[r - wide:r + wide + 1, c - wide:c + wide + 1, 1] = 0
    img_left[r - wide:r + wide + 1, c - wide:c + wide + 1, 2] = 255

    img_cyc = np.copy(cyc_img)
    r = cyc_p[1]
    c = cyc_p[0]
    #print(f"cyclopean image pixel row:{r} col:{c}")
    img_cyc[r - wide:r + wide + 1, c - wide:c + wide + 1, 0] = 255  # RGB layers in the image
    img_cyc[r - wide:r + wide + 1, c - wide:c + wide + 1, 1] = 0
    img_cyc[r - wide:r + wide + 1, c - wide:c + wide + 1, 2] = 255

    nrow = 1
    ncol = 3
    fig_img, ax_img = plt.subplots(nrow, ncol)
    ax_img[0].imshow(img_left)
    ax_img[1].imshow(img_right)
    ax_img[2].imshow(img_cyc)
    return


def mark_image_center(some_image, wide=5):

    '''
    :param image is right eye
    :param location where the match was found with the left eye
    '''

    img_new = np.copy(some_image)
    r = 256
    c = 256
    img_new[r - wide:r + wide + 1, c - wide:c + wide + 1, 0] = 255  # RGB layers in the image
    img_new[r - wide:r + wide + 1, c - wide:c + wide + 1, 1] = 0
    img_new[r - wide:r + wide + 1, c - wide:c + wide + 1, 2] = 255
    return img_new

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
    7 = agent_pos           # to compute left and right eye use baseline
    8 = [lefteye rotation, righteye rotation]
    9 = robot_head_neck_rotation
    :type sal_processed_dir: directory which should contain 11 sal-processed pickled file
    :return: a tuple consisting of an image list, a salmap list
    :rtype: tuple
    '''

    my_data = read_file(filename)
    astate = []
    if my_data is not None:
        imageL = my_data[0][0]
        imageR = my_data[0][1]
        pointsL = my_data[4][0]
        pointsR = my_data[4][1]
        astate.append(my_data[6])
        astate.append(my_data[7])
        astate.append(my_data[8][0])
        astate.append(my_data[8][1])

    return imageL, imageR, pointsL, pointsR, astate


def get_corresponding_point_in_bw(left_image, right_image, left_point):

    height = left_image.shape[0]
    width = left_image.shape[1]
    win_span = 2  # window span is 2*win_span + 1
    fix_row = left_point[0]
    fix_col = left_point[1]
    r1 = fix_row - win_span
    r2 = fix_row + win_span
    c1 = fix_col - win_span
    c2 = fix_col + win_span
    print(f"Fixation point in Left Image {fix_row}, {fix_col}")

    if (0 <= r1 < height and 0 <= r2 < height and 0 <= c1 < width and 0 <= c2 < width):
        patch = left_image[r1:r2 + 1, c1:c2 + 1, :]
        patch = rgb2gray(patch)
        plt.imshow(patch)
        plt.show()
        right_gray_image = rgb2gray(right_image)
        result = match_template(right_gray_image, patch, pad_input=True)
        loc = np.unravel_index(np.argmax(result), result.shape)
        print(f"Match Point {loc[0]}, {loc[1]}")
        match_image = display_image_showing_match(left_image, left_point, right_image, loc)
        plt.imshow(match_image)
        plt.show()
        return loc
    else:
        print(f"Patch boundaries {r1}: {r2} or {c1}:{c2} not within left_image shape {left_image.shape}")
        return None


def get_corresponding_point_in_color(left_image, right_image, left_point, win_span = 3):

    height = left_image.shape[0]
    width = left_image.shape[1]
    fix_row = left_point[0]
    fix_col = left_point[1]
    r1 = fix_row - win_span
    r2 = fix_row + win_span
    c1 = fix_col - win_span
    c2 = fix_col + win_span
    print(f"Fixation point in Left Image {fix_row}, {fix_col}")

    '''
    nrow = 1
    ncol = 1
    fig2, ax2 = plt.subplots(nrow, ncol)
    nrow = 1
    ncol = 1
    fig3, ax3 = plt.subplots(nrow, ncol)
    '''

    if (0 <= r1 < height and 0 <= r2 < height and 0 <= c1 < width and 0 <= c2 < width):
        patch = left_image[r1:r2 + 1, c1:c2 + 1, :]
        #ax2.imshow(patch)
        result = match_template(right_image, patch, pad_input=True)
        loc = np.unravel_index(np.argmax(result), result.shape)
        print(f"Match Point {loc[0]}, {loc[1]}")
        #match_image = display_image_showing_match(left_image, left_point, right_image, loc)
        #ax3.imshow(match_image)
        #plt.show()
        return loc[:2]
    else:
        print(f"Patch boundaries {r1}: {r2} or {c1}:{c2} not within left_image shape {left_image.shape}")
        return None


def get_disparity_map(left_image, right_image, my_row, win_span = 3):

    height = left_image.shape[0]
    width =  left_image.shape[1]

    start_row = win_span
    end_row = height - win_span  # it is 1 more than the end

    if (win_span <= my_row < height - win_span):  # from 3 to 508
        pass
    else:
        print(f"Invalid Row {my_row}. Specify a row from {win_span} to {height - win_span - 1}")

    my_image = right_image[my_row-win_span:my_row+win_span+1,:,:]

    disparity_map = []

    start_col = win_span
    end_col = width - win_span  # it is 1 more than the end

    for i in range(win_span):
        disparity_map.append(0)
    for j in range(start_col, end_col):
        patch = left_image[my_row - win_span:my_row + win_span + 1, j - win_span:j + win_span + 1, :]
        result = match_template(my_image, patch, pad_input=True)
        loc = np.unravel_index(np.argmax(result), result.shape)
        print(f"Left Image Pixel ({my_row},{j}) matches Right Image Pixel ({loc[0]}, {loc[1]})")
        if loc[0] == win_span:
            disparity_map.append(j - loc[1])
        else:
            disparity_map.append(0)
    for i in range(win_span):
        disparity_map.append(0)

    return np.asarray(disparity_map)



def display_common_image(img1, map1, img2, map2):
    map1_2D = np.copy(map1)
    map1_3D = np.dstack([map1_2D, map1_2D, map1_2D])
    new_img1 = img1 * map1_3D

    map2_2D = np.copy(map2)
    map2_3D = np.dstack([map2_2D, map2_2D, map2_2D])
    new_img2 = img2 * map2_3D
    return [new_img1,new_img2]


def map_common_pixels_from_rotation_only(R1, R2, focal_distance, width, height):
    '''
    Reference for rotation is Habitat World Coordinates
    :param R1 Rotation in quaterion from WCS of the starting position - reference pixel map
    :param R2 Rotation in quaterion from WCS to the next position - new pixel map
    :param focal_distance: the frame z coordinate is equal to -focal_distance
    :param width: sensor width pixels
    :param height: sensor height pixels
    :returns common a binary map of common pixel position in the starting frame and the new frame
    '''

    R = R2.inverse()*R1     #Rotation from any image frame to start image frame
    w = int(width/2)
    h = int(height/2)

    common = 0
    new = 0
    fd_squared = focal_distance*focal_distance
    ref_map = np.zeros((height,width),dtype=int)
    new_map = np.zeros((height,width),dtype=int)

    my_rotation = quaternion.as_rotation_matrix(R)
    for xpos in range(width):               # xpos in start image - column
        for ypos in range(height):          # ypos in start image - row
            # x, y for origin at the center of the frame and computing the unit vector
            x = xpos - w
            y = h - ypos
            v = np.array([x, y, -focal_distance])
            uvector = v / np.linalg.norm(v)

            # rotate the uvector
            new_vector = my_rotation.dot(uvector.T)

            ux = new_vector[0]
            uy = new_vector[1]
            uz = new_vector[2]
            # calculate angles that the unit vector makes with z axis and with xz plane
            uxz = np.sqrt(ux * ux + uz * uz)
            theta = np.arcsin(ux / uxz)  # z is never zero, theta is the angle wrt to y
            phi = np.arcsin(uy)  # x is the angle wrt to z
            # compute x,y (z = -focal length)
            xval = int(round(focal_distance * np.tan(theta)))
            xz = np.sqrt(fd_squared+(xval*xval))
            yval = int(round(xz * np.tan(phi)))
            # convert to top left origin
            xnew = xval + w
            ynew = h - yval
            #original_pixel = (ypos,xpos)
            #corresponding_pixel = (ynew, xnew)
            if 0 <= xnew < width and 0<= ynew < height:
                #common_pixels.append([original_pixel, corresponding_pixel])
                #other_image_exclusion.append(corresponding_pixel)
                ref_map[ypos, xpos] = 1
                new_map[ynew, xnew] = 1
                common +=1
            else:
                new +=1
                #only_start_image_pixels.append(original_pixel)        # in numpy array order
    return ref_map, new_map


if __name__ == "__main__":


    oreo_in_habitat = agent_oreo.agent_oreo(agent_oreo.scene, agent_oreo.dest_folder, agent_oreo.pyBfolder,
                                            depth_camera=True, loc_depth_cam='c', foveation=False,phys=False)
    my_file = datafile_location + datafile_name
    l_image, r_image, l_points, r_points, agent_state = get_image_and_salmap(my_file)

    #shift_at_center = get_corresponding_point_in_color(l_image, r_image, (256, 256))
    # compute depth and limit computations to sensors as depth camera is not required further)
    #depth_map = oreo_in_habitat.set_and_capture_depth(agent_state[1:3])
    #depth_map_by_baseline = depth_map/agent_oreo.eye_separation
    #oreo_in_habitat.num_sensors = 2

    '''
    row_num = 255
    dmap = get_disparity_map(l_image,r_image,row_num)
    print(f"Depth map shape ------> {dmap.shape}")
    row_depth = np.divide(depth_map_by_baseline[row_num,:].reshape(dmap.shape),dmap)
    

    l_image_with_points = display_image_with_salpoints(l_image,l_points)
    r_image_with_points = display_image_with_salpoints(r_image, r_points)

    nrow = 1
    ncol = 3
    fig1, ax = plt.subplots(nrow, ncol)
    ax[0].imshow(l_image_with_points)
    ax[1].imshow(r_image_with_points)
    ax[2].imshow(depth_map)
    '''
    '''
    oreo_in_habitat.restore_state(oreo_in_habitat.setup_agent_state(agent_state))
    left_img = oreo_in_habitat.my_images["left_rgb_sensor"][..., 0:3]
    right_img = oreo_in_habitat.my_images["right_rgb_sensor"][..., 0:3]
    cyc_img = oreo_in_habitat.my_images["depth_sensor"][..., 0:3]

    nrow = 1
    ncol = 3
    fig_1, ax1 = plt.subplots(nrow, ncol)
    ax1[0].imshow(left_img)
    ax1[1].imshow(right_img)
    ax1[2].imshow(cyc_img)

    v1 = np.array([0.0, 0.0, -1.0])
    v2 = oreo_in_habitat.compute_uvector_for_image_point(434,152)
    cyc_pt = (434, 152)
    head_angle = np.arccos(np.dot(v1, v2))  # Angle of rotation for axis-angle representation
    head_axis = np.cross(v1, v2)

    head_rot = head_angle * head_axis
    head_rot_quat = quaternion.from_rotation_vector(head_rot)
    oreo_in_habitat.rotate_head_neck_primary_position(head_rot_quat)

    nrow = 1
    ncol = 3
    fig_2, ax2 = plt.subplots(nrow, ncol)
    left_nimg = oreo_in_habitat.my_images["left_rgb_sensor"][..., 0:3]
    right_nimg = oreo_in_habitat.my_images["right_rgb_sensor"][..., 0:3]
    cyc_nimg = oreo_in_habitat.my_images["depth_sensor"][..., 0:3]
    ax2[0].imshow(left_nimg)
    ax2[1].imshow(right_nimg)
    ax2[2].imshow(cyc_nimg)

    print(f"New Head direction in Axis  = {head_axis} Angle = {head_angle}")
    print(f"New Head direction in Axis Angle = {head_rot} in Quaternion = {head_rot_quat}")

    '''
    cyc_img = oreo_in_habitat.my_images["depth_sensor"][..., 0:3]
    cyc_pt = (434, 152)

    for i in l_points:
        # left image point row is i[2], col is i[3]
        right_image_point = get_corresponding_point_in_color(l_image, r_image, i[2:])
        #right_image_point = get_corresponding_point_in_bw(l_image, r_image, i[2:])
        display_image_showing_match(l_image, i[2:], r_image, right_image_point, cyc_img, cyc_pt, wide=4)
        if right_image_point is not None:
            binocular_fixated_imgs, a_state, fd = \
                oreo_in_habitat.capture_binocular_fixation_images(agent_state, i[2:], right_image_point)
            #self.my_images["left_rgb_sensor"][..., 0:3], self.my_images["right_rgb_sensor"][..., 0:3]]
            if binocular_fixated_imgs is not None:
                nrow = 1
                ncol = 3
                fig_2, ax2 = plt.subplots(nrow, ncol)
                left_nimg = oreo_in_habitat.my_images["left_rgb_sensor"][..., 0:3]
                right_nimg = oreo_in_habitat.my_images["right_rgb_sensor"][..., 0:3]
                cyc_nimg = oreo_in_habitat.my_images["depth_sensor"][..., 0:3]
                ax2[0].imshow(mark_image_center(oreo_in_habitat.my_images["left_rgb_sensor"][..., 0:3]))
                ax2[1].imshow(mark_image_center(oreo_in_habitat.my_images["depth_sensor"][..., 0:3]))
                ax2[2].imshow(mark_image_center(oreo_in_habitat.my_images["right_rgb_sensor"][..., 0:3]))
                plt.show()
                done = True
                '''
                #self.my_images, self.get_current_state(), self.focal_distance
                lefteye_orn = a_state.sensor_states["left_rgb_sensor"].rotation
                p_left = np.array([i[2], i[3], -fd])
                p_left_wcs = quaternion.as_rotation_matrix(lefteye_orn).dot(p_left.T)
                righteye_orn = a_state.sensor_states["right_rgb_sensor"].rotation
                p_right = np.array([right_image_point[0], right_image_point[1], -fd])
                p_right_wcs = quaternion.as_rotation_matrix(righteye_orn).dot(p_right.T)
                print(f"Left point is {p_left_wcs} and Right point is {p_right_wcs}")

                figure, axis = plt.subplots(3, 2)
                axis[0, 0].imshow(binocular_fixated_imgs[0])
                axis[0, 1].imshow(binocular_fixated_imgs[1])
                array_dim = binocular_fixated_imgs[0].shape
                hei = array_dim[0]
                wid = array_dim[1]
                lefteye_orn = a_state.sensor_states["left_rgb_sensor"].rotation
                righteye_orn = a_state.sensor_states["right_rgb_sensor"].rotation
                mapL, mapR = map_common_pixels_from_rotation_only(lefteye_orn,righteye_orn,fd, wid, hei)
                common_img = display_common_image(binocular_fixated_imgs[0],mapL,binocular_fixated_imgs[1],mapR)
                axis[1,0].imshow(common_img[0])
                axis[1,1].imshow(common_img[1])
                plt.show()

                diff_imag = binocular_fixated_imgs[0] - binocular_fixated_imgs[1]
                comm_image = (binocular_fixated_imgs[0] + binocular_fixated_imgs[1])/2
                merged_img = rgb2gray(diff_imag)
                merged_img = np.where(merged_img < 0.01, 1, 0)
                axis[2, 0].imshow(comm_image)
                axis[2, 1].imshow(merged_img)
                #plt.show()
                '''
                pass
            else:
                print(f"No image - cannot saccade to the new point")
        else:
            print(f"Patch not fully within left image")

    a = 5