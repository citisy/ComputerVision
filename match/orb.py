"""Oriented FAST and Rotated BRIEF"""

import cv2


def orb(query, match):
    orb = cv2.ORB_create(1000)
    img1 = cv2.imread(query)  # queryImage
    kp1, des1 = orb.detectAndCompute(img1, None)

    img2 = cv2.imread(match)  # matchImage
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return img1, kp1, img2, kp2, matches


if __name__ == '__main__':
    query = 'img/lena512color.png'
    match = 'img/detected.png'
    img1, kp1, img2, kp2, matches = orb(query, match)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    # cv2.imwrite('img/orb.png', img3)
    cv2.imshow('orb', img3)
    cv2.waitKeyEx()
    cv2.destroyAllWindows()
