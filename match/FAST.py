"""Features From Accelerated Segment Test
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html
"""


import numpy as np
import cv2


class NpFAST:
    def detected(self, query, t=10):
        mask = np.zeros((7, 7), dtype=int)
        mask[0, 2:5] = mask[-1, 2:5] = mask[2:5, 0] = mask[2:5, -1] = \
            mask[1, 1] = mask[-2, 1] = mask[1, -2] = mask[-2, -2] = 1
        # mask[0, 3] = mask[-1, 3] = mask[3, 0] = mask[3, -1] = 1  # compute more quickly
        corner = []
        for i in range(query.shape[0] - 7):
            for j in range(query.shape[1] - 7):
                x = query[i:i + 7, j:j + 7].copy()
                p = x[3, 3]
                x = mask * x
                d, s, b = 0, 0, 0
                for a in x:
                    for b in a:
                        if b != 0:
                            if b <= p - t:
                                d += 1
                            if p - t < b < p + t:
                                s += 1
                            if p + t <= b:
                                b += 1
                if d >= 12 or b >= 12:
                    corner.append((j + 3, i + 3))
        return corner


def cv_FAST(query, t=10):
    fast = cv2.FastFeatureDetector_create(threshold=t)  # FastFeatureDetector for python2

    kp = fast.detect(query, None)

    return kp


if __name__ == '__main__':
    query = cv2.imread('img/lena512color.png')
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)  # change to gray image

    model = NpFAST()
    corner = model.detected(query_gray, t=50)
    for i in corner:
        cv2.circle(query, i, 3, (50, 50, 180), 2)
    cv2.imwrite('img/FAST.png', query)

    # kp = cv_FAST(query_gray, t=50)
    # img = cv2.drawKeypoints(query, kp, query, color=(255, 0, 0))   # only use under cv2 version, 3.3.0
    # cv2.imshow('de', query)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
