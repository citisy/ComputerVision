import cv2
import numpy as np

img = cv2.imread("train_img/query.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.05)

# dst = cv2.dilate(dst,None)

img[dst > 0.015 * dst.max()] = [0, 0, 255]

cv2.imshow("harris_points", img)

cv2.waitKeyEx(0)
cv2.destroyAllWindows()