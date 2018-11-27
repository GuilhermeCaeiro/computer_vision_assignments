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

def draw_points(image, keypoins, matches):
    if len(keypoins) > 0:
        for keypoint in keypoins:
            #print(keypoint)
            cv2.circle(image, tuple(keypoint), 2, (0,0,255), -1)
    
    if len(matches) > 0:
        for match in matches:
            cv2.circle(image, tuple(match), 2, (0,255,255), -1)

    return image

def generate_matching_image(image1, image2, matches):
    matching_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], image1.shape[2]), dtype="uint8")
    print(matching_image.shape)
    print(image1.shape, image2.shape)

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





def transform(homography, base_image, second_image):
    base_image_width = base_image.shape[1]
    base_image_height = base_image.shape[0]

    second_image_width = second_image.shape[1]
    second_image_height = second_image.shape[0]

    homography_inverse = np.linalg.inv(homography)

    


    # get the mapping for the second image's corners on the base image's (reference) plane
    second_image_top_left = homography_inverse * np.matrix([[0, 0, 1]]).T
    second_image_bottom_left = homography_inverse * np.matrix([[0, second_image_height, 1]]).T
    second_image_bottom_right = homography_inverse * np.matrix([[second_image_width, second_image_height, 1]]).T
    second_image_top_right = homography_inverse * np.matrix([[second_image_width, 0, 1]]).T

    print("Corners", second_image_top_left, second_image_bottom_left, second_image_bottom_right, second_image_top_right)

    #(line_one * (1/line_one[0, 2]))
    print("dividing by x3")
    second_image_top_left = second_image_top_left * (1 / second_image_top_left[2, 0])
    second_image_bottom_left = second_image_bottom_left * (1 / second_image_bottom_left[2, 0])
    second_image_bottom_right = second_image_bottom_right * (1 / second_image_bottom_right[2, 0])
    second_image_top_right = second_image_top_right * (1 / second_image_top_right[2, 0])

    print("Corners", second_image_top_left, second_image_bottom_left, second_image_bottom_right, second_image_top_right)

    min_x = min(second_image_top_left[0, 0], second_image_bottom_left[0, 0])
    min_y = min(second_image_top_left[1, 0], second_image_top_right[1, 0])
    max_x = max(second_image_top_right[0, 0], second_image_bottom_right[0, 0])
    max_y = max(second_image_bottom_left[1, 0], second_image_bottom_right[1, 0])

    print("Mins and maxes", min_x, min_y, max_x, max_y)

    x_left_displacement = (0 - min_x) if min_x < 0 else 0 
    x_right_displacement = (max_x - base_image_width) if max_x > base_image_width else 0 
    y_top_displacement = (0 - min_y) if min_y < 0 else 0
    y_bottom_displacement = (max_y - base_image_height) if max_y > base_image_height else 0

    """displacements = {
        "x_left_displacement": x_left_displacement,
        "x_right_displacement": x_right_displacement,
        "y_top_displacement": y_top_displacement,
        "y_bottom_displacement": y_bottom_displacement
    }"""

    print("Displacements", x_left_displacement, x_right_displacement, y_top_displacement, y_bottom_displacement)

    new_base_image_width = base_image_width + x_left_displacement + x_right_displacement
    new_base_image_height = base_image_height + y_top_displacement + y_bottom_displacement

    print("New image size", new_base_image_width, new_base_image_height)

    # with the reference corners for the two images, write each pixel of the base 
    # image in a new image, that comprises all the corners of the two images,
    # and then write the second image in that plane, mapping each of its pixels

    new_base_image_matrix = np.zeros((new_base_image_height, new_base_image_width, 3), dtype=np.int)

    for x in range(0, base_image_width):
        for y in range(0, base_image_height):
            target_row = int(y_top_displacement + y)
            target_column = int(x_left_displacement + x)
            try:
                new_base_image_matrix[target_row, target_column] = base_image[y, x] 
            except:
                pass
            #print(x, y, target_column, target_row, base_image[y, x])

    for x in range(0, second_image_width):
        for y in range(0, second_image_height):
            mapping = homography_inverse * np.matrix([[x, y, 1]]).T
            
            # the mapping coordinate is then adjusted and the value of the current pois is written
            #mapping = mapping * (1 / mapping[2, 0])
            try: 
                if list(new_base_image_matrix[y_top_displacement + mapping[1, 0], x_left_displacement + mapping[0, 0]]) == [0, 0, 0]:
                    new_base_image_matrix[y_top_displacement + mapping[1, 0], x_left_displacement + mapping[0, 0]] = second_image[y, x]
                #new_base_image_matrix[y_top_displacement + mapping[1, 0], x_left_displacement + mapping[0, 0]] = second_image[y, x]
            except:
                pass
    #print(new_base_image_matrix)
    return new_base_image_matrix
            
#def is_in_bounds(point, x_left_displacement, x_right_displacement, y_top_displacement, y_bottom_displacement):
#    if (x_left_displacement + point[0, 0]) >= x_left_displacement and (y_top_displacement + mapping[1, 0]) >= y_top_displacement and 
#    return 


#files = ["m_img1.png", "m_img2.png", "m_img3.png"]
#files = ["alcimg1.png", "alcimg2.png", "alcimg3.png", "alcimg4.png"]#, "alcimg5.png", "alcimg6.png", "alcimg7.png",]
files = ["alcimg1.png", "alcimg2.png", "alcimg3.png", "alcimg4.png", "alcimg5.png", "alcimg6.png", "alcimg7.png",]
#files = ["P_20181111_023936.jpg", "P_20181111_023940.jpg", "P_20181111_023943.jpg", "P_20181111_023946.jpg", "P_20181111_023950.jpg", "P_20181111_023954.jpg", "P_20181111_023956.jpg"]
#files = ["IMG-20181109-WA0003_1.jpg", "IMG-20181109-WA0004_2.jpg", "IMG-20181109-WA0005_3.jpg", "IMG-20181109-WA0001_4.jpg", "IMG-20181109-WA0002_5.jpg"]

base_image = None

for i in range(len(files) - 2, -1, -1):
    print("Stitching images " + str(i) + " and " + str(i + 1))

    img1 = base_image

    if img1 is None:
        print("No base image found. Using LAST image in the files' array.")
        img1 = cv2.imread(files[i + 1])
        img1 = imutils.resize(img1, width=400)

    img2 = cv2.imread(files[i])
    img2 = imutils.resize(img2, width=400)

    #temp = img1
    #img1 = img2
    #img2 = temp

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

    #print(img2.shape[1])
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

    #result = cv2.warpPerspective(img1, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    #cv2.imwrite("result-p1.jpg", result)
    #result[0:img2.shape[0], 0:img2.shape[1]] = img2
    #cv2.imwrite("result-p2.jpg", result)

    result = transform(homography, img1, img2).astype('uint8')
    cv2.imshow("Base Image", result)
    #output_image_array = transform(homography, img2, output_image_array)
    #cv2.imshow("img2", output_image_array)


    #vis = drawMatches(img1, img2, img1_points, img2_points, matches, status)

    image1_with_points = draw_points(img1, img1_kps, img1_points)
    image2_with_points = draw_points(img2, img2_kps, img2_points)

    matching_image = generate_matching_image(image2_with_points, image1_with_points, matches)

    matching_image_with_lines = draw_lines(matching_image, img2_points, img1_points, matches)

    cv2.imshow("Matches", matching_image_with_lines)


    base_image = result

    #cv2.imshow("Image " + str(i), img1)
    #cv2.imshow("Image " + str(i + 1), img2)
    ##cv2.imshow("Keypoint Matches", vis)
    #cv2.imshow("Result", result)
    cv2.waitKey(0)




