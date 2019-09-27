"""Scale-Invariant Feature Transform
"""

import cv2
import numpy as np

img = cv2.imread('train_img/query.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

siftDetector = cv2.xfeatures2d.SIFT_create(100)     # todo: This algorithm is patented and is excluded in this configuration
kp, res = siftDetector.detectAndCompute(gray, None)
print(len(kp))
print(res)

img = cv2.drawKeypoints(img, outImage=img, keypoints=kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("sift", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

