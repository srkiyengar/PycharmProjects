import pickle
import os
import saliency
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import math
import random

def fix_sal_mistake():
    my_dir = "/Users/rajan/PycharmProjects/saliency/saliency_map/results/"
    for root, dirs, files in os.walk(my_dir):
        for filename in files:
            if "-sal-processed" in filename:
                p_salfile = my_dir + filename
                data = read_file(p_salfile)
                salmapL = data[1][0]
                salmapR = data[1][1]
                both_salmap = [salmapL, salmapR]
                d = filename.find("-sal-processed")
                new_name = filename[0:d+4]
                newfile = my_dir + new_name
                try:
                    with open(newfile, "wb") as f:
                        pickle.dump(both_salmap, f)
                except:
                    print(f"Failure: To open and save saliency file {newfile}")

def save_all_images_and_salmap():
    my_dir = "/Users/rajan/PycharmProjects/saliency/saliency_map/results/"
    salmaps = []
    images = []
    imagesfile = None
    salmapsfile = None
    for root, dirs, files in os.walk(my_dir):
        for filename in files:
            if "-sal-processed" in filename:
                p_salfile = my_dir + filename
                try:
                    data = read_file(p_salfile)
                except:
                    print(f"Failure: To read file {p_salfile}")
                    break
                images.append(data[0][0])
                salmaps.append(data[1][0])
                d = filename.find('RGB')
                if d == -1:
                    d = filename.find('rgb')
                imagesfile = filename[0:d] + "-images"
                salmapsfile = filename[0:d] + "-salmaps"

    if imagesfile is not None:
        try:
            fname = my_dir + imagesfile
            with open(fname, "wb") as f:
                pickle.dump(images, f)
        except:
            print(f"Failure: To open and save saliency file {fname}")

    if salmapsfile is not None:
        try:
            fname = my_dir + salmapsfile
            with open(fname, "wb") as f:
                pickle.dump(salmaps, f)
        except:
            print(f"Failure: To open and save saliency file {fname}")


def read_file(myfilename):
    try:
        with open(myfilename, "rb") as f:
            data = pickle.load(f)
    except IOError as e:
        print("Failure: Loading pickle file {}".format(myfilename))
        exit(1)
    return data

def read_images_and_salmap():

    my_dir = "/Users/rajan/PycharmProjects/saliency/saliency_map/results/"
    imagesfile = my_dir + "van-gogh-room.glb^2021-09-26-08-58-31-images"
    salmapsfile = my_dir + "van-gogh-room.glb^2021-09-26-08-58-31-salmaps"

    images = read_file(imagesfile)
    nrow = 2
    ncol = 5
    Images, ax1 = plt.subplots(nrow, ncol)
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
    plt.show()

    Salmaps, ax2 = plt.subplots(nrow, ncol)
    salmaps = read_file(salmapsfile)
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




def view_images_and_salmap():
    my_dir = "/Users/rajan/PycharmProjects/saliency/saliency_map/old/"
    #my_dir = "/Users/rajan/PycharmProjects/saliency/saliency_map/results/"

    salmaps = []
    images = []
    for root, dirs, files in os.walk(my_dir):
        for filename in files:
            if "-sal-processed" in filename:
                p_salfile = my_dir + filename
                try:
                    data = read_file(p_salfile)
                except:
                    print(f"Failure: To read file {p_salfile}")
                    break
                images.append(data[0][0])
                salmaps.append(data[1][0])

    nrow = 2
    ncol = 5
    Images, ax1 = plt.subplots(nrow, ncol)
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


    Salmaps, ax2 = plt.subplots(nrow, ncol)
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





class image_set(object):

    def __init__(self,start_image):
        self.start_imagefile = start_image
        self.start_image_salfile = start_image + "-sal"
        self.start_image_fixation_file = self.start_image_salfile + "-processed"
        self.fixation_images_file = self.start_image_fixation_file + "-images"
        self.fixation_images_salfile = self.fixation_images_file + "-sal-emsemble"

    def view_start_image(self):
        image_data = read_file(self.start_imagefile)
        rows = 1
        cols = 2
        fig, axes = plt.subplots(rows, cols, num = "Start Left and Right Images")
        axes[0].imshow(image_data[8][0])
        axes[1].imshow(image_data[8][1])
        pass

    def view_start_image_and_salmap(self):
        data_from_salfile = read_file(self.start_image_fixation_file)
        start_imageL = data_from_salfile[0][0]
        start_imageR = data_from_salfile[0][1]
        start_image_salmapL = data_from_salfile[1][0]
        start_image_salmapR = data_from_salfile[1][1]
        rows = 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, num = "Start Image and Saliency map")


        axes[0, 0].imshow(start_imageL)
        #axes[0, 1].imshow(start_imageL, alpha=0.2)
        m = axes[0, 1].matshow((start_image_salmapL), alpha=0.5, cmap=plt.cm.RdBu)
        axes[0, 1].matshow(start_imageL, alpha=0.2, cmap=plt.cm.RdBu)
        fig.colorbar(m, ax=axes[0, 1])

        axes[0, 0].axis('off')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(start_imageR)
        #axes[1, 1].imshow(start_imageR, alpha=0.2)
        n = axes[1, 1].matshow((start_image_salmapR), alpha=0.5, cmap=plt.cm.RdBu)
        axes[1, 1].matshow((start_imageR), alpha=0.2, cmap=plt.cm.RdBu)
        fig.colorbar(n, ax=axes[1, 1])

        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

    def view_fixation_points(self):
        data_from_processed_file = read_file(self.start_image_fixation_file)
        salmap_with_fixationsL = data_from_processed_file[3][0]
        salmap_with_fixationsR = data_from_processed_file[3][1]
        start_imageL = data_from_processed_file[0][0]
        start_imageR = data_from_processed_file[0][1]
        start_image_salmapL = data_from_processed_file[1][0]
        start_image_salmapR = data_from_processed_file[1][1]

        rows = 3
        cols = 2
        fig, axes = plt.subplots(rows, cols, num = "Image, Saliency map and fixations")


        axes[0, 0].imshow(start_imageL)
        axes[1, 0].imshow(start_imageL, alpha=0.2)
        m = axes[1, 0].matshow((start_image_salmapL), alpha=0.5, cmap=plt.cm.RdBu)
        fig.colorbar(m, ax=axes[1, 0])
        axes[2, 0].imshow(start_imageL, alpha=0.2)
        axes[2, 0].matshow((salmap_with_fixationsL), alpha=0.5, cmap=plt.cm.RdBu)

        axes[0, 1].imshow(start_imageR)
        axes[1, 1].imshow(start_imageR, alpha=0.2)
        n = axes[1, 1].matshow((start_image_salmapR), alpha=0.5, cmap=plt.cm.RdBu)
        fig.colorbar(n, ax=axes[1, 1])
        axes[2, 1].imshow(start_imageR, alpha=0.2)
        axes[2, 1].matshow((salmap_with_fixationsR), alpha=0.5, cmap=plt.cm.RdBu)

        axes[0, 0].axis('off')
        axes[0, 1].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')


    def view_fixation_centered_images(self, which = "left"):
        my_ensemble = saliency.sal_ensemble(self.fixation_images_file)
        left_images, right_images = my_ensemble.get_images()
        if which == "left":
            images = left_images
        else:
            images = right_images

        rows = 2
        cols = 5
        fig, axes = plt.subplots(rows, cols, num = "Images from fixations")
        for r in range(rows):
            for c in range(cols):
                img = images[c + r*cols]
                if img is not None:
                    i = r*5+c
                    axes[r, c].set_title(f"image {i+1}")
                    axes[r, c].imshow(images[c + r*cols])
                axes[r, c].axis('off')

        plt.show()



def write_file(myfilename,output):
    try:
        with open(myfilename, "wb") as f:
            pickle.dump(output, f)
            print(f"Saved new saliency file {myfilename}")

    except IOError as e:
        print(f"Failure: To open/write file {myfilename}")

#starting_point_image = "/Users/rajan/PycharmProjects/saliency/saliency_map/results/van-gogh-room.glb^2021-07-22-10-21-10RGB"
starting_point_image = \
    "/Users/rajan/PycharmProjects/saliency/saliency_map/results/van-gogh-room.glb^2021-09-26-08-58-31RGB9"
#starting_point_image = \
#    "/Users/rajan/PycharmProjects/saliency/saliency_map/results/van-go-2021-07-22-10-12-02RGB/van-gogh-room.glb^2021-07-22-10-12-02RGB0"


if __name__ == "__main__":
    #fix_sal_mistake()
    #read_images_and_salmap()
    #save_all_images_and_salmap()
    view_images_and_salmap()

    '''
    my_set = image_set(starting_point_image)
    my_set.view_start_image()
    my_set.view_start_image_and_salmap()
    my_set.view_fixation_points()
    my_set.view_fixation_centered_images()
    '''




    '''
    a = np.arange(12.).reshape(4,3,1)
    b = np.arange(12.).reshape(3, 4, 1)
    c = np.tensordot(a, b, axes=([1, 0], [0, 1]))

    my_image_object = saliency.sal_ensemble(processed_image)
    images = my_image_object.get_images()

    total_images = len(images[0])
    row = 2
    ncols = int(total_images/row)
    sal_img_data = read_file(sal_images)
    fig, axes = plt.subplots(row, ncols)

    for r in range(row):
        for c in range(ncols):
            index = r*ncols + c
            img = images[0][index]
            if img is not None:
                axes[r, c].imshow(img,alpha=0.2)
                axes[r, c].matshow((sal_img_data[0][index]), alpha=0.5, cmap=plt.cm.RdBu)
                axes[r, c].axis('off')
            else:
                axes[r, c].set_axis_off()
                #axes[r, c].axis('off')
    i = 3
    saved_salmap = sal_img_data[0][i]
    img = images[0][i]
    plt.gca().imshow(img, alpha=0.2)
    m = plt.gca().matshow((saved_salmap), alpha=0.5, cmap=plt.cm.RdBu)
    plt.colorbar(m)
    plt.title('Saliency prediction')
    plt.axis('off');




    my_data = read_file(processed_image)
    imgL_data = my_data[0]
    imgR_data = my_data[1]
    '''

    '''

    total_images = len(imgL_data)
    row = 2
    ncols = int(total_images/row)

    fig, axes = plt.subplots(row, ncols)

    for i in range(row):
        for j in range(ncols):
            index = i*ncols+j
            if imgL_data[index][1] is not None:
                axes[i,j].imshow(imgL_data[index][1][4])
                axes[i,j].set_title(str(index))
    plt.show()


    consolidated.append(my_comparison.images)
    consolidated.append(my_comparison.salmap)
    consolidated.append(my_comparison.recon_image)
    consolidated.append(ref_point_list)
    consolidated.append(my_array)
    
    my_images = my_data[1]
    total_images = len(my_images)
    row = 2
    #ncols = int(total_images/row)
    ncols = 2
    fig, axes = plt.subplots(row,ncols)

    axes[0, 0].imshow(my_images[0])
    axes[0, 0].set_title('1')
    axes[0, 1].imshow(my_images[1])
    axes[0, 1].set_title('2')

    axes[1, 0].imshow(my_images[2])
    axes[1, 0].set_title('3')
    axes[1, 1].imshow(my_images[3])
    axes[1, 1].set_title('4')

    plt.show()
    '''

