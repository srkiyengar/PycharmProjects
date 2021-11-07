import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

max = 11        #Number of images Start + 10

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

def get_images_and_salmaps(dir_path, sal_processed_dir):
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


    image_list = []
    salmap_list = []
    d = sal_processed_dir.find('RGB0')
    common_name = sal_processed_dir[0:d + 3]
    scene = common_name
    common_name = "van-gogh-room.glb^" + common_name[3:]
    #common_name = "skokloster-castle.glb^"+ common_name[3:]

    for i in range(max):
        filename = common_name + str(i) + '-sal-processed'
        filename = dir_path + '/' + sal_processed_dir + '/' + filename
        my_data = read_file(filename)
        if my_data is not None:
            image_list.append(my_data[0][0])
            salmap_list.append(my_data[1][0])
        else:
            print(f"Error - directory {sal_processed_dir} filename {filename}")
            image_list.append(None)
            salmap_list.append(None)
    return image_list, salmap_list, scene


def display_salmaps(salmaps):

    salmap = salmaps.pop(0)

    nrow = 1
    ncol = 2

    for i, j in enumerate(salmaps):
        I, ax = plt.subplots(nrow, ncol)
        ax[0].axis('off')
        ax[0].set_title("Start Fixation Heatmap")
        ax[0].imshow(salmap)
        ax[1].axis('off')
        if j is not None:
            jacccard_dist = jaccard_similarity_coefficient_vectors(salmap,j)
            my_label = f"Fixation {i + 1} heatmap"
            ax[1].imshow(j)
            ax[1].set_title(my_label)
            I.suptitle(f'Jaccard Coefficient {jacccard_dist}', fontsize=12)
            plt.show()
    return


def display_10_salmaps(salmaps,scene):

    nrow = 2
    ncol = 5
    I, ax = plt.subplots(nrow, ncol)
    I.suptitle(f'Scene {scene}', fontsize=12)
    print(f"Display 10 salmaps: Number of salmaps: {len(salmaps)}")
    for i, j in enumerate(salmaps):
        x = i // ncol
        y = i % ncol
        if i == 0:
            start_salmap = j
        if i > 9:
            break
        #ax[x, y].axis('off')
        if j is not None:
            my_label = f"Fixation {i}"
            ax[x, y].imshow(j)
            ax[x, y].set_title(my_label)
            jaccard_coe = jaccard_similarity_coefficient_vectors(start_salmap, j)
            b_coe = bhattacharyaa_coefficient(start_salmap,j)
            sh_entropy, cr_entropy, _ = salmap_entropy(j,true_map=start_salmap)
            if i == 0:
                entropy = sh_entropy
            KL_div = cr_entropy - entropy
            ax[x, y].set_xlabel(
                    f'Jacc.Index {jaccard_coe:.3f}\nB. Coe. = {b_coe:.3f}\n Image Entropy = {sh_entropy:.3f}\n'
                    f'Cross Entropy = {cr_entropy:.3f}\n KL Div = {KL_div:.3f}')


    plt.show()
    return


def compute_distances_for_10_salmaps(salmaps):

    jc_list = []
    bc_list = []
    shannonE_list = []
    crossE_list = []
    kl_list = []
    max_list = []

    print(f"Compute distances for 10 salmaps: Number of salmaps: {len(salmaps)}")
    for i, j in enumerate(salmaps):
        if i == 0:
            start_salmap = j
        if i > 9:
            break
        if j is not None:
            jaccard_coe = jaccard_similarity_coefficient_vectors(start_salmap, j)
            b_coe = bhattacharyaa_coefficient(start_salmap,j)
            sh_entropy, cr_entropy, _ = salmap_entropy(j,true_map=start_salmap)
            if i == 0:
                entropy = sh_entropy
            KL_div = cr_entropy - entropy
            maxval = np.amax(j)
            jc_list.append(jaccard_coe)
            bc_list.append(b_coe)
            shannonE_list.append(sh_entropy)
            crossE_list.append(cr_entropy)
            kl_list.append(KL_div)
            max_list.append(maxval)
        else:
            jc_list.append(None)
            bc_list.append(None)
            shannonE_list.append(None)
            crossE_list.append(None)
            kl_list.append(None)
            max_list.append(None)
    return [jc_list,bc_list,shannonE_list,crossE_list,kl_list,max_list]


def display_images_salmaps_and_distances(scene, images, salmaps, distances):

    jc = distances[0]
    bc = distances[1]
    sE = distances[2]
    cE = distances[3]
    kl = distances[4]
    mval = distances[5]

    nrow = 2
    ncol = 5
    fig1, ax1 = plt.subplots(nrow, ncol)
    fig2, ax2 = plt.subplots(nrow, ncol)
    fig1.suptitle(f'Scene {scene} Images 0 to 4', fontsize=12)
    fig2.suptitle(f'Scene {scene} Images 5 to 9', fontsize=12)
    images.pop()
    salmaps.pop()

    print(f"Display Images Salmaps and Distance: Number of images: {len(images)}\n Salmaps {len(salmaps)}")

    for i, j in enumerate(zip(images,salmaps)):
        if j[0] is not None and j[1] is not None:
            x = i // ncol
            y = i % ncol
            if x == 0:
                ax1[0, y].imshow(j[0])
                ax1[0, y].set_title(f"Fix {y}")
                ax1[1, y].imshow(j[1])
                ax1[1, y].set_xlabel(f'Jacc.Index {jc[i]:.3f}\nB. Coe. = {bc[i]:.3f}\nImage Entropy = {sE[i]:.3f}\n'
                                     f'Cross Entropy = {cE[i]:.3f}\nKL Div = {kl[i]:.3f}\nMax. Sal Val = {mval[i]:.6f}')
            else:
                ax2[0, y].imshow(j[0])
                ax2[0, y].set_title(f"Fix {ncol + y}")
                ax2[1, y].imshow(j[1])
                ax2[1, y].set_xlabel(f'Jacc.Index {jc[i]:.3f}\nB. Coe. = {bc[i]:.3f}\nImage Entropy = {sE[i]:.3f}\n'
                                     f'Cross Entropy = {cE[i]:.3f}\nKL Div = {kl[i]:.3f}\nMax. Sal Val = {mval[i]:.6f}')
    #plt.show()


def display_distances(distances, scene):

    jc = distances[0]
    bc = distances[1]
    #sE = distances[2]
    #cE = distances[3]
    kl = distances[4]
    #mval = distances[5]

    fig = plt.figure(f"Stat. Distance - {scene}")
    ax = fig.add_subplot(1, 1, 1)
    xdata = range(len(jc))
    ax.plot(xdata, jc, color='tab:blue', label='Jaccard Coefficient')
    ax.plot(xdata, bc, color='tab:orange', label='Bhattacharya Coefficient')
    ax.plot(xdata, kl, color='tab:green', label='KL Divergence')
    ax.set_xlabel("Fixations")
    fig.suptitle(f"Stat. Distances for {scene}", fontsize=12)
    ax.legend()
    plt.show()


def display_avg_diff(salmaps, scene):

    fig, ax = plt.subplots(1, 2)
    fig.suptitle(f'Scene {scene}', fontsize=12)
    first_salmap = salmaps.pop(0)
    if first_salmap is not None:
        ax[0].imshow(first_salmap)
        my_shape = first_salmap.shape
        avg_salmap = np.zeros(my_shape, dtype=float)
        first_entropy, cross_entropy, _ = salmap_entropy(first_salmap,true_map=first_salmap)
        jc = jaccard_similarity_coefficient_vectors(first_salmap, first_salmap)
        bc = bhattacharyaa_coefficient(first_salmap, first_salmap)
        ax[0].set_xlabel(f'Image Entropy = {first_entropy:.3f}')
        ax[0].set_title("Fixation 0 - Heatmap")
        ax[0].set_xlabel(f'Jacc.Index {jc:.3f}\nB. Coe. = {bc:.3f}\nImage Entropy = {first_entropy:.3f}\n'
                         f'Cross Entropy = {cross_entropy:.3f}')
    else:
        print(f"display_avg_diff: Salmap 0 is null")
        return
    count = 0
    for j in salmaps:
        if j is not None:
            avg_salmap = np.add(avg_salmap,j)
            count +=1
    avg_salmap = avg_salmap/count
    print(f"Number of Salmap images averaged {count}")
    jc = jaccard_similarity_coefficient_vectors(first_salmap, avg_salmap)
    bc = bhattacharyaa_coefficient(first_salmap, avg_salmap)
    avg_entropy, cross_entropy, _ = salmap_entropy(avg_salmap, true_map=first_salmap)
    ax[1].imshow(avg_salmap)
    ax[1].set_title("Avg Fix - Heatmap")
    ax[1].set_xlabel(f'Jacc.Index {jc:.3f}\nB. Coe. = {bc:.3f}\nEntropy = {avg_entropy:.3f}\n'
                     f'Cross Entropy = {cross_entropy:.3f}')
    plt.show()


def display_images_and_salmaps (images, salmaps):

    fig, ax = plt.subplots(1,2)
    my_label = f"Start Image"
    img = images.pop(0)
    salmap =salmaps.pop(0)
    ax[0].imshow(img)
    ax[1].imshow(salmap)
    ax[0].set_title("Image")
    ax[1].set_title("Heatmap")
    ax[0].axis('off')
    ax[1].axis('off')

    nrow = 2
    ncol = 5
    I1, ax1 = plt.subplots(nrow, ncol)
    for i, j in enumerate(images):
        x = i // ncol
        y = i % ncol
        if i > 9:
            break
        ax1[x, y].axis('off')
        if j is not None:
            my_label = f"Image {i + 1}"
            ax1[x, y].imshow(j)
            ax1[x, y].set_title(my_label)

    I2, ax2 = plt.subplots(nrow, ncol)
    for i, j in enumerate(salmaps):
        x = i // ncol
        y = i % ncol
        if i > 9:
            break
        ax2[x, y].axis('off')
        if j is not None:
            my_label = f"Salmap {i + 1}"
            ax2[x, y].imshow(j)
            ax2[x, y].set_title(my_label)
    plt.show()
    return

def create_salmap_pickle(s_list, my_path, dir_name):
    sal_pickle_file = my_path + "/" + dir_name + "/" + dir_name + "-sal-list"
    heatmap_list = []
    for heatmap in s_list:
        a = heatmap.flatten()
        heatmap_list.append(a)

    write_file(sal_pickle_file,heatmap_list)


def compute_histogram(map_list,scenename):
    # the histogram of the data

    nrows = 2
    ncols = 5

    highest = 0
    lowest = 0

    for i in map_list:
        if i is not None:
            minval = np.min(i[np.nonzero(i)])
            maxval = np.max(i[np.nonzero(i)])
            if maxval > highest:
                highest = maxval
            if minval > lowest:
                lowest = minval


    fig, ax = plt.subplots(nrows, ncols)
    fig.suptitle(f'{scenename}: Histogram of Heatmaps', fontsize=12)
    print(f"lowest = {lowest} and highest = {highest}")
    lowest = 0.00002
    for i,j in enumerate(map_list):
        if j is not None:
            x = i // ncols
            y = i % ncols
            fix = x * ncols + y
            if fix == ncols * nrows:
                break
            ax[x, y].hist(np.ravel(j), 10, range=(lowest, highest), facecolor='g', alpha=0.75)
            #ax[x, y].axis('off')
            ax[x, y].set_title(f'Fixation {fix}')
            # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
            # plt.xlim(40, 160)
            # plt.ylim(0, 0.03)
            # plt.grid(True)
    plt.show()
    return


def jaccard_similarity_coefficient_vectors(a1, a2):
    '''
    :param a1, a2: two normalized numpy vectors of same dimensions
    :returns the coefficient, a value between 0 and 1
    '''

    b = np.minimum(a1, a2)
    c = np.maximum(a1, a2)

    d = np.sum(b)
    e = np.sum(c)
    f = d/e

    return f

def bhattacharyaa_coefficient(a1, a2):
    '''
    :param a1, a2: two normalized numpy vectors of same dimensions
    :returns the coefficient, a value between 0 and 1
    '''

    b = np.sum(np.sqrt(a1 * a2))
    return b

def salmap_entropy(pred_map, true_map=None):

    '''
    marg = list(filter(lambda p: p > 0, np.ravel(my_map)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    '''
    shannon_entropy = 0.0
    cross_entropy = 0.0
    if pred_map is None:
        return None, None, 0
    row, col = pred_map.shape
    count = 0
    if true_map is not None:
        for i in range(row):
            for j in range(col):
                pred_q = pred_map[i, j]
                true_p = true_map[i, j]
                if pred_q > 0:
                    log_pred_q = np.log2(pred_q)
                    shannon_entropy += -pred_q * log_pred_q
                    count += 1
                    if true_p > 0:
                        cross_entropy += -true_p * log_pred_q

        return shannon_entropy, cross_entropy, count
    else:
        for i in range(row):
            for j in range(col):
                pred_q = pred_map[i, j]
                if pred_q > 0:
                    log_pred_q = np.log2(pred_q)
                    shannon_entropy += -pred_q * log_pred_q
                    count += 1

        return shannon_entropy, None, count


my_dir = "/Users/rajan/PycharmProjects/saliency/saliency_data"


if __name__ == "__main__":

    k = "van2021-10-24-02-00-25RGB0"
    path = "/Users/rajan/PycharmProjects/saliency/saliency_data"
    img_list, smap_list, _ = get_images_and_salmaps(path, k)
    ldp = smap_list[0]
    size = 256

    A1 = np.ones(ldp.shape)
    B1 = np.ones((size, size))

    shape = np.add(A1.shape, B1.shape)
    A2 = np.zeros(shape)
    B2 = np.zeros(shape)
    A2[:A1.shape[0], :A1.shape[1]] = A1
    B2[:B1.shape[0], :B1.shape[1]] = B1


    #k = "van2021-10-24-02-00-25RGB0"
    #k = "van2021-10-24-23-36-04RGB0"
    #k = "van2021-10-24-23-38-45RGB0"   #more uniform

    '''
    path = "/Users/rajan/PycharmProjects/saliency/saliency_data"
    display_salmaps(smap_list)
    create_salmap_pickle(smap_list, path, k)
    if img_list is not None:
        display_images_and_salmaps(img_list, smap_list)
    '''


    for path, subdirs, files in os.walk(my_dir):
        for k in subdirs:
            img_list, smap_list, scene = get_images_and_salmaps(path, k)
            print(f"Number of images: {len(img_list)}\nNumber of Salmaps: {len(smap_list)}")
            temp = compute_histogram(smap_list, scene)
            #create_salmap_pickle(smap_list, path, k)
            if img_list is not None:
                #isplay_salmaps(smap_list)
                #display_images_and_salmaps(img_list, smap_list)
                distances = compute_distances_for_10_salmaps(smap_list)
                display_distances(distances,scene)
                display_images_salmaps_and_distances(scene, img_list, smap_list, distances)
                display_10_salmaps(smap_list,scene)
                display_avg_diff(smap_list, scene)
            pass
    pass
