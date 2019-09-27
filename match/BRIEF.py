"""Binary Robust Independent Elementary Features"""

import numpy as np
import cv2

img = cv2.imread('img/lena512color.png')

# Initiate STAR detector
star = cv2.xfeatures2d.SIFT_create()

# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(img,None)

# compute the descriptors with BRIEF
keypoints, des = brief.compute(img, kp)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        color=(0, 0, 255))

cv2.imshow('sift', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# keypoints, descriptor = sift.detectAndCompute(gray, None)
# img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#                         color=(0, 0, 255))
#
# cv2.imshow('sift', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()