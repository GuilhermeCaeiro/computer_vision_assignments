import numpy as np
import scipy.linalg
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from sys import argv
import math
import cv2

files = ["img1.png", "img2.png"]
base_image = None

for i in range(len(files) - 1):
    img1 = base_image

    if img1 == None:
        img1 = cv2.imread(files[i])

    img2 = cv2.imread(files[i + 1])

    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst1 = cv2.cornerHarris(gray, 2, 3, 0.04)

    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst2 = cv2.cornerHarris(gray, 2, 3, 0.04)

    print(dst, dst.shape)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img1[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img1)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()