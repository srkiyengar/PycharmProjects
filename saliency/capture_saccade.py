
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
import agent_oreo as ao
import math

smooth_results = "/Users/rajan/PycharmProjects/saliency/smooth_results"

myfile = "redball_drop_frames"
#myfile = "moving_object_frames"

def find_max_and_index(array_2d):
    '''
    :param array_2d: is a saliency map .
    :type array_2d: ndarray of rows x columns
    :return: maximum value, its row, and its column
    :rtype: tuple
    np.max returns the maximum value in array_2d and the == operator on array_2d produces a boolean ndarray and the
    non-zero returns two arrays which provide the row, col values of the occurance of max value. I am taking the first
    r,c location and running with it.
    '''

    result = np.nonzero(array_2d == np.amax(array_2d))
    r = result[0][0]
    c = result[1][0]
    return np.amax(array_2d), r, c

def modified_hough(vel_mag, phase_list, image_data):
    #image_point_values.append(image location: [i,j], ratio of dI/|G| at [i,j], theta G at [i,j]])
    x = len(vel_mag)
    y = len(phase_list)
    min_angle = np.pi/8
    vel_phase = []
    for m in phase_list:
        vel_phase.append(m*min_angle)
    tolerance = 0.05
    accumulator = np.zeros((x,y), dtype=int)
    #velomulator = np.zeros((x,y), dtype=float)
    for i in image_data:
        #r = i[0][0]
        #c = i[0][1]

        vG = i[1]
        tG = i[2]
        #print(f"row: {r} col: {c} dI/G = {vG}\n")

        phase = []
        for vphase in vel_phase:
            phase.append((np.cos(tG - vphase)))
        for c,p in enumerate(phase):
            for r, vmag in enumerate(vel_mag):
                v = -vmag*p         #-|v|cos(thetaG-thetaV)
                #print(f"comparing vG: {vG} with magV: {v} abs_diff: {abs(vG-v)}")
                if abs(vG-v) < tolerance:
                    accumulator[r, c] += 1
    print(f"vPhase={vel_phase}")
    print(f"vmag = {vel_mag}")
    count, r, c = find_max_and_index(accumulator)
    return count, vel_mag[r], vel_phase[c]


def read_file(somefilename):
    try:
        with open(somefilename, "rb") as f:
            data = pickle.load(f)
            return data
    except IOError as e:
        print("Failure: Loading pickle file {}".format(somefilename))
        return None




myfilename = smooth_results + "/" + myfile
image_data = read_file(myfilename)      #list of lists 0) frame numbers and 1) frames




'''
if image_data is not None:
    for i in zip(image_data[0], image_data[1]):
        ao.display_single_frame(i[0], i[1])
        k = cv2.waitKey(0)
        if k == ord('q'):
            break

'''
img1_rgb = image_data[1][6]
img1_bw= cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2GRAY)
img1 = cv2.GaussianBlur(img1_bw, (7,7), 0)
sobelx1 = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=3) # x
sobely1 = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=3)  # y
dG1 = (np.sqrt(np.square(sobelx1) + np.square(sobely1)))/2

"""
print(f"Intensity for row213 from 175 to 215 \ndx1: {sobelx1[213,175:215]}\n"
      f"dx2: {sobelx2[213,175:215]}\n"
      f"dx3: {sobelx3[213,175:215]}")

fig1, ax = plt.subplots(1, 2)
ax[0].imshow(img1_bw)
ax[1].imshow(img1)
plt.show()

"""




img2_rgb = image_data[1][7]
img2_bw = cv2.cvtColor(img2_rgb, cv2.COLOR_BGR2GRAY)
img2 = cv2.GaussianBlur(img2_bw, (7,7), 0)
sobelx2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)  # x
sobely2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)  # y
dG2 = (np.sqrt(np.square(sobelx2) + np.square(sobely2)))/2


#fig1, ax = plt.subplots(1, 2)
#ax[0].imshow(img1_bw)
#ax[1].imshow(img2_bw)
#print(f"Intensity for row213 from 175 to 215 \nimg1: {img1_bw[213,175:215]} \nimg2: {img2_bw[213,175:215]}")
#print(f"Intensity at (213, 191) in imag1:{img1[213,197]} and img2:{img2[213,191]}")

with np.errstate(divide='ignore'):
    tantheta1 = sobely1/sobelx1
    tantheta1[sobelx1 == 0] = 0
    theta1 = np.arctan(tantheta1)
    theta1 = theta1

    tantheta2 = sobely2 / sobelx2
    tantheta2[sobelx2 == 0] = 0
    theta2 = np.arctan(tantheta2)
    theta2 = theta2

nrow = 2
ncol = 4
fig2, ax = plt.subplots(nrow, ncol)
ax[0,0].imshow(img1_rgb)
ax[0,1].imshow(img1)
ax[0,2].imshow(dG1)
ax[0,3].imshow(theta1)

ax[1,0].imshow(img2_rgb)
ax[1,1].imshow(img2_bw)
ax[1,2].imshow(dG2)
ax[1,3].imshow(theta2)

dG3 = np.absolute(dG1 - dG2)
dG3[dG3 > 50] = 0
count_dG3 = np.count_nonzero(dG3)

dR = dG3.copy()
dR[dR != 0] = 1
dR[dR != 1] = 0     # image points which are excluded from computation

#dI = dI*dG3

dI = img2-img1
dI[dI<20] = 0       # Assuming the range is 0 to 255

nrow = 1
ncol = 3
fig3, ax = plt.subplots(nrow, ncol)
ax[0].imshow(dG3)
ax[1].imshow(dI)



dtheta = np.absolute(theta1-theta2)


dtheta[dtheta > 0.1] = 0
dtheta[dtheta > 0] = 1

dE = dR*dtheta # All points with 0 will be excluded from image computation
num_points = np.count_nonzero(dE)
print(f"Number of pixels for comuting velocity {num_points}")

dI = dI*dE
theta1 = theta1*dE
dG1 = dG1*dE
ax[2].imshow(dI)

#ax[3,0].imshow(dE)
#ax[3,1].imshow(dG1)

plt.show()

r = dG1.shape[0]
c = dG1.shape[1]

x = []
y = []

image_point_values = []
for i in range(r):
    for j in range(c):
        if dG1[i,j] != 0 and dI[i,j] != 0:
            #vG = 2*dI[i,j]/dG1[i,j]           #Sobel operator multiplier removed.
            vG = dI[i, j] / dG1[i, j]
            y.append(vG)
            x.append(theta1[i,j])
            image_point_values.append([[i,j],vG,theta1[i,j]])
            #print(f"dI: {dI[i,j]}, dG1: {dG1[i,j]}, thetaG: {theta1[i,j]} vG: {vG}\n")
            #print(f"r: {i}, c: {j}: dI: {dI[i, j]}, dG1: {dG1[i, j]}, thetaG: {theta1[i, j]} vG: {vG}")
magnitudes = [3,4,5,6,7,8,9,10,11,12,13,14,15]
#phases = [0,1,2,3,4,5,6,7,8,-1,-2,-3,-4,-5,-6,-7]
phases = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15]
val, velocity,direction = modified_hough(magnitudes, phases, image_point_values)

print(f"Magnitude = {val}, Velocity = {velocity}, phase = {direction}")
#plt.scatter(x, y)
#plt.scatter(dtheta1, vG)
#plt.show()



#dG1[dG1 < 0.1] = 0
#dG4[dG4 < 1] = -1.0

pass