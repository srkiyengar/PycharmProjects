import pickle
import cv2


def read_file(myfilename):
    try:
        with open(myfilename, "rb") as f:
            data = pickle.load(f)
            return data
    except IOError as e:
        print("Failure: Loading pickle file {}".format(myfilename))
        return None



# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)

my_file = \
    "/Users/rajan/PycharmProjects/saliency/saliency_data/van2021-10-09-18-02-10RGB0/van2021-10-09-18-02-10RGB0-sal-list"

start_processed = \
    "/Users/rajan/PycharmProjects/saliency/saliency_data/van2021-10-24-23-38-45RGB0/van-gogh-room.glb^2021-10-24-23-38-45RGB0-sal-processed"

mouseX = None
mouseY = None
my_count = 1

def get_mouse_2click(event,x,y,flags,param):
    global mouseX, mouseY, my_count
    #print(f"Inside callback event = {event}")
    if event != 0:
        print(f"{my_count} Button Clicked: {event} x:{x}, y:{y}")
        mouseX = x
        mouseY = y
        my_count +=1


if __name__ == "__main__":

    print(f"{cv2.EVENT_MOUSEMOVE}")

    with open(start_processed, "rb") as f:
        filedata = pickle.load(f)

    img = filedata[0][0]

    # reading the image
    #img = cv2.imread('lena.jpg', 1)

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
    cv2.setMouseCallback('image', get_mouse_2click)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()


    pass
