import numpy as np
import scipy.linalg
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from sys import argv
import math
import cv2
import imutils
import copy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# Based on http://www.mathworks.com/help/vision/examples/structure-from-motion-from-two-views.html

def draw_points(image, keypoins, matches):
    image = copy.deepcopy(image)

    if len(keypoins) > 0:
        for keypoint in keypoins:
            #print(keypoint)
            cv2.circle(image, tuple(keypoint), 2, (0,0,255), -1)
    
    if len(matches) > 0:
        for match in matches:
            cv2.circle(image, tuple(match), 2, (0,255,255), -1)

    return image

def generate_matching_image(image1, image2, matches):
    matching_image = np.zeros((image1.shape[0], image1.shape[1] * 2, image1.shape[2]), dtype="uint8")
    print(matching_image.shape)

    matching_image[0:image1.shape[0], 0:image1.shape[1]] = image1
    matching_image[0:image2.shape[0], image1.shape[1]:] = image2

    print(matching_image.shape)

    #for match in matches:



    return matching_image

def draw_lines(matching_image, image1_points, image2_points, matches):
    for point_left, point_right in zip(image1_points, image2_points): # where "order" means the order number, like 1st, 7th, etc
        point_left = tuple([int(i) for i in point_left])
        #point_right = tuple([int(i) for i in point_right])
        #point_left = (int(image1_points[order_left][0]), int(image1_points[order_left][1]))
        point_right = (int(point_right[0]) + int(matching_image.shape[1] / 2), int(point_right[1]))
        cv2.line(matching_image, point_left, point_right, (255, 255, 0), 1)

    return matching_image



image1 = cv2.imread("DSC_0047.jpg")
image2 = cv2.imread("DSC_0048.jpg")

image1 = imutils.resize(image1, width=1000)
image2 = imutils.resize(image2, width=1000)

# obtaining matching points

orb = cv2.ORB_create()
image1_keypoints = orb.detect(image1, None)
image1_keypoints, image1_features = orb.compute(image1, image1_keypoints)

orb = cv2.ORB_create()
image2_keypoints = orb.detect(image2, None)
image2_keypoints, image2_features = orb.compute(image2, image2_keypoints)

image1_keypoints = np.float32([kp.pt for kp in image1_keypoints])
image2_keypoints = np.float32([kp.pt for kp in image2_keypoints])

print("Number of keypoints", len(image1_keypoints), len(image2_keypoints))
print("Keypoint sample", image1_keypoints[0])

matcher = cv2.DescriptorMatcher_create("BruteForce")
rawMatches = matcher.knnMatch(image1_features, image2_features, 2)

print("Number of raw matches", len(rawMatches))
print(rawMatches[0])

matches = []
ratio = 0.8
reprojThresh = 4.0
confidence = 0.99

for match in rawMatches:
    #print(match)
    # ensure the distance is within a certain ratio of each
    # other (i.e. Lowe's ratio test)
    if len(match) == 2 and match[0].distance < match[1].distance * ratio:
        print(match[0].trainIdx, match[0].queryIdx)
        matches.append((match[0].trainIdx, match[0].queryIdx))

print("Length of matches list", len(matches))
print("Matches", matches)

image1_points = np.float32([image1_keypoints[i] for (_, i) in matches])
image2_points = np.float32([image2_keypoints[i] for (i, _) in matches])

print(image1_points)
print(image2_points)

"""
image1_with_points = draw_points(image1, image1_keypoints, image1_points)
image2_with_points = draw_points(image2, image2_keypoints, image2_points)
matching_image = generate_matching_image(image1_with_points, image2_with_points, matches)
matching_image_with_lines = draw_lines(matching_image, image1_points, image2_points, matches)
"""

# Camera Matrix
"""K = np.matrix([
    [(60.0 * 6000) / 23.5, 0, 2000],
    [0, (60.0 * 6000) / 15.6, 3000],
    [0, 0, 1]
])"""

"""K = np.matrix([
    [(60.0 * 4000) / 15.6, 0, 2000],
    [0, (60.0 * 6000) / 23.5, 3000],
    [0, 0, 1]
])"""

K = np.matrix([
    [(60.0 * 4000) / 23.5, 0, 2000],
    [0, (60.0 * 6000) / 15.6, 3000],
    [0, 0, 1]
])

"""K = np.matrix([
    [(60.0 * 6000) / 23.5, 0, 3000],
    [0, (60.0 * 4000) / 15.6, 2000],
    [0, 0, 1]
])"""


#F, masks = cv2.findEssentialMat(image1_points, image2_points, cv2.FM_RANSAC, ransacReprojThreshold = reprojThresh, confidence = confidence)
E, masks = cv2.findEssentialMat(image1_points, image2_points, cameraMatrix = K, method = cv2.FM_RANSAC, threshold = 1.0, prob = 0.999)

print(E, masks)

mask_as_integer_list = [] 

for i in range(len(masks)):
    if masks[i] == 1:
        mask_as_integer_list.append(i)

print(mask_as_integer_list)

image1_inlier_keypoints = image1_points[mask_as_integer_list]
image2_inlier_keypoints = image2_points[mask_as_integer_list]

print(len(image1_inlier_keypoints), image1_inlier_keypoints)

print(len(image1_inlier_keypoints), len(image1_points))

image1_with_points = draw_points(image1, image1_points, image1_inlier_keypoints)
image2_with_points = draw_points(image2, image2_points, image2_inlier_keypoints)

matching_image = generate_matching_image(image1_with_points, image2_with_points, matches)

matching_image_with_lines = draw_lines(matching_image, image1_inlier_keypoints, image2_inlier_keypoints, matches)


retval, R, t, mask, triangulatedPoints = cv2.recoverPose(E, image1_points, image2_points, cameraMatrix = K, distanceThresh = 1.0, mask = masks)
#retval, R, t, mask, triangulatedPoints = cv2.recoverPose(E, image1_points, image2_points, cameraMatrix = K, distanceThresh = 1.0)
#retval, R, t, mask, triangulatedPoints = cv2.recoverPose(E, image1_inlier_keypoints, image2_inlier_keypoints, cameraMatrix = K, distanceThresh = 1.0)
print("\n\n\n")
print(retval) 
print(R)
print(t)
print(mask)
print(triangulatedPoints)

print(len(triangulatedPoints[0]))

triangulated_points = []

for i in range(len(triangulatedPoints[0])):
    if mask[i][0] == 0:
        continue

    w = triangulatedPoints[3][i]
    a_triangulated_point = [triangulatedPoints[0][i], triangulatedPoints[1][i], triangulatedPoints[2][i], w]

    #print(a_triangulated_point)

    a_triangulated_point = [triangulatedPoints[0][i]/w, triangulatedPoints[1][i]/w, triangulatedPoints[2][i]/w, w/w]

    #print(a_triangulated_point)

    triangulated_points.append(a_triangulated_point)

print(len(triangulated_points), "\n", triangulated_points)


fig = pyplot.figure()
ax = Axes3D(fig)
#ax.scatter(triangulatedPoints[0], triangulatedPoints[1], triangulatedPoints[2])
ax.scatter([i[0] for i in triangulated_points], [i[1] for i in triangulated_points], [i[2] for i in triangulated_points])
pyplot.show()


cv2.imshow("Image 1", image1_with_points)
cv2.imshow("Image 2", image2_with_points)

cv2.imshow("Matches", matching_image_with_lines)

cv2.waitKey(0)
