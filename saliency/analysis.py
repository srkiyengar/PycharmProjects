import pickle
import os
import saliency
import matplotlib.pyplot as plt




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




if __name__ == "__main__":

    read_images_and_salmap()