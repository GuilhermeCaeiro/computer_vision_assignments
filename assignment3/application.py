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

files = ["img1.png", "img2.png"]
base_image = None

for i in range(len(files) - 1):
    img1 = base_image

    if img1 == None:
        img1 = cv2.imread(files[i])

    img2 = cv2.imread(files[i + 1])

    img1 = imutils.resize(img1, width=400)
    img2 = imutils.resize(img2, width=400)

    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst1 = cv2.cornerHarris(gray, 2, 3, 0.04)

    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst2 = cv2.cornerHarris(gray, 2, 3, 0.04)

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

    result = cv2.warpPerspective(img1, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    vis = drawMatches(img1, img2, img1_points, img2_points, matches, status)

    cv2.imshow("Image " + str(i), img1)
    cv2.imshow("Image " + str(i + 1), img2)
    #cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)




