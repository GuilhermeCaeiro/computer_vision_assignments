import numpy as np
import scipy.linalg
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from sys import argv
import math
import cv2
import imutils

# Based on https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

def drawMatches(img1, img2, img1_points, img2_points, matches, status):
    # initialize the output visualization image
    (img1_height, img1_width) = img1.shape[:2]
    (img2_height, img2_width) = img2.shape[:2]
    vis = np.zeros((max(img1_height, img2_height), img1_width + img2_width, 3), dtype="uint8")
    vis[0:img1_height, 0:img1_width] = img1
    vis[0:img2_height, img1_width:] = img2

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(img1_points[queryIdx][0]), int(img1_points[queryIdx][1]))
            ptB = (int(img2_points[trainIdx][0]) + img1_width, int(img2_points[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # return the visualization
    return vis


def transform(homography, deformed_image_array, output_image_array = None):
    homography_matrix = homography
    homography_matrix_inverse = np.linalg.inv(homography_matrix)

    deformed_image_width = deformed_image_array.shape[1]
    deformed_image_height = deformed_image_array.shape[0]

    print(deformed_image_array.shape, deformed_image_width, deformed_image_height)
    print(deformed_image_array[0])
    print(deformed_image_array[0][0])

    # Defining points of interest
    deformed_x_min = 0
    deformed_x_max = deformed_image_width
    deformed_y_min = 0
    deformed_y_max = deformed_image_height

    pre_corner0 = homography_matrix * np.matrix([[deformed_x_min, deformed_y_min, 1]]).T
    pre_corner1 = homography_matrix * np.matrix([[deformed_x_min, deformed_y_max, 1]]).T
    pre_corner2 = homography_matrix * np.matrix([[deformed_x_max, deformed_y_max, 1]]).T
    pre_corner3 = homography_matrix * np.matrix([[deformed_x_max, deformed_y_min, 1]]).T

    print("pre", pre_corner0, pre_corner1, pre_corner2, pre_corner3)

    pre_corner0 = np.matrix([[int(pre_corner0[0,0] / pre_corner0[2,0]), int(pre_corner0[1,0] / pre_corner0[2,0]), int(pre_corner0[2,0] / pre_corner0[2,0])]]).T
    pre_corner1 = np.matrix([[int(pre_corner1[0,0] / pre_corner1[2,0]), int(pre_corner1[1,0] / pre_corner1[2,0]), int(pre_corner1[2,0] / pre_corner1[2,0])]]).T
    pre_corner2 = np.matrix([[int(pre_corner2[0,0] / pre_corner2[2,0]), int(pre_corner2[1,0] / pre_corner2[2,0]), int(pre_corner2[2,0] / pre_corner2[2,0])]]).T
    pre_corner3 = np.matrix([[int(pre_corner3[0,0] / pre_corner3[2,0]), int(pre_corner3[1,0] / pre_corner3[2,0]), int(pre_corner3[2,0] / pre_corner3[2,0])]]).T

    #pre_corner0 = pre_corner0.astype(int)
    #pre_corner1 = pre_corner1.astype(int)
    #pre_corner2 = pre_corner2.astype(int)
    #pre_corner3 = pre_corner3.astype(int)

    print("pre", pre_corner0, pre_corner1, pre_corner2, pre_corner3)

    pre_corners = [pre_corner0, pre_corner1, pre_corner2, pre_corner3]

    x_min = math.inf
    y_min = math.inf
    x_max = - math.inf
    y_max = - math.inf

    for pre_corner in pre_corners:
        if pre_corner[0, 0] < x_min:
            x_min = pre_corner[0, 0]

        if pre_corner[0, 0] > x_max:
            x_max = pre_corner[0, 0]

        if pre_corner[1, 0] < y_min:
            y_min = pre_corner[1, 0]

        if pre_corner[1, 0] > y_max:
            y_max = pre_corner[1, 0]

    print("mins", x_min, y_min, x_max, y_max)

    total_x = x_max - x_min
    total_y = y_max - y_min

    print("totals", total_x, total_y, total_y/total_x)

    # output
    output_width = 768
    output_height = int((total_y/total_x) * output_width)

    #output_width = int(total_x)
    #output_height = int(total_y)

    #print("dims", output_width, output_height)

    step_x = total_x / output_width
    step_y = total_y / output_height

    #print("step sizes", step_x, step_y)
    if output_image_array is None:
        output_image_array = np.zeros((output_height, output_width, 3), dtype=np.int)

    #print(type(output_image_array))
    print(output_image_array.shape)

    print("border points", x_min, y_min)
    print(homography_matrix_inverse * np.matrix([[x_min, y_min, 1]]).T)
    #print("\n")

    for x in range(0, output_width):
        for y in range(0, output_height):
            deformed_point = homography_matrix_inverse * np.matrix([[x_min + int(x * step_x), y_min + int(y * step_y), 1]]).T

            try:
                unscaled_deformed_point = [int(deformed_point[0,0] / deformed_point[2,0]), int(deformed_point[1,0] / deformed_point[2,0])]

                if((unscaled_deformed_point[0] < 0 or unscaled_deformed_point[0] >= deformed_image_width) or (unscaled_deformed_point[1] < 0 or unscaled_deformed_point[1] >= deformed_image_height)):
                    continue

                output_image_array[y, x] = deformed_image_array[unscaled_deformed_point[1], unscaled_deformed_point[0]]
                
            except:
                pass#print("error")


    #print(output_image_array[0])
    #print(output_image_array[0][0])
    print(output_image_array.shape)

    return output_image_array









files = ["m_img1.png", "m_img2.png", "m_img3.png"]
#files = ["alcimg1.png", "alcimg2.png", "alcimg3.png"]#, "alcimg4.png"]#, "alcimg5.png", "alcimg6.png", "alcimg7.png",]

base_image = None

for i in range(len(files) - 1):
    print("Stitching images " + str(i + 1) + " and " + str(i + 2))

    img1 = base_image

    if img1 is None:
        print("No base image found. Using first image in the files array.")
        img1 = cv2.imread(files[i])

    img2 = cv2.imread(files[i + 1])

    img1 = imutils.resize(img1, width=400)
    img2 = imutils.resize(img2, width=400)

    temp = img1
    img1 = img2
    img2 = temp

    #gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #gray = np.float32(gray)
    #dst1 = cv2.cornerHarris(gray, 2, 3, 0.04)

    #gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #gray = np.float32(gray)
    #dst2 = cv2.cornerHarris(gray, 2, 3, 0.04)

    # detect and extract features from the image
    orb = cv2.ORB_create()
    img1_kps = orb.detect(img1,None)
    img1_kps, img1_features = orb.compute(img1, img1_kps)

    #print(img1_kps, img1_features)

    orb = cv2.ORB_create()
    img2_kps = orb.detect(img2,None)
    img2_kps, img2_features = orb.compute(img2, img2_kps)

    #img1_descriptor = cv2.xfeatures2d.SIFT_create()
    #(img1_kps, img1_features) = descriptor.detectAndCompute(img1, None)

    #img2_descriptor = cv2.xfeatures2d.SIFT_create()
    #(img2_kps, img2_features) = descriptor.detectAndCompute(img2, None)

    img1_kps = np.float32([kp.pt for kp in img1_kps])
    img2_kps = np.float32([kp.pt for kp in img2_kps])

    # maching points
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(img1_features, img2_features, 2)
    
    matches = []
    ratio = 0.75
    reprojThresh = 4.0

    for match in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(match) == 2 and match[0].distance < match[1].distance * ratio:
            matches.append((match[0].trainIdx, match[0].queryIdx))

    homography = None
    status = None

    print("Number of matches ", len(matches))
    if len(matches) > 4:
        print("More than 4 matches")
        # construct the two sets of points
        img1_points = np.float32([img1_kps[i] for (_, i) in matches])
        img2_points = np.float32([img2_kps[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        homography, status = cv2.findHomography(img1_points, img2_points, cv2.RANSAC, reprojThresh)

    print(homography)
    #print(status)

    result = cv2.warpPerspective(img2, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    cv2.imwrite("result-p1.jpg", result)
    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    cv2.imwrite("result-p2.jpg", result)

    #output_image_array = transform(homography, img1, None)
    #cv2.imshow("img1", output_image_array)
    #output_image_array = transform(homography, img2, output_image_array)
    #cv2.imshow("img2", output_image_array)


    #vis = drawMatches(img1, img2, img1_points, img2_points, matches, status)

    base_image = result

    cv2.imshow("Image " + str(i), img1)
    cv2.imshow("Image " + str(i + 1), img2)
    #cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)




