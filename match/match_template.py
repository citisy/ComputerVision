"""暴力匹配"""

import numpy as np
import cv2

# CV_TM_SQDIFF = 0
# CV_TM_SQDIFF_NORMED = 1
# CV_TM_CCORR = 2
# CV_TM_CCORR_NORMED = 3
# CV_TM_CCOEFF = 4
# CV_TM_CCOEFF_NORMED = 5


class NpMt:
    def detected(self, query, match, method=None):
        if method is None:
            method = self.cv_tm_sqdiff
        res = np.zeros((query.shape[0] - match.shape[0], query.shape[1] - match.shape[1]))
        for i in range(query.shape[0] - match.shape[0]):
            for j in range(query.shape[1] - match.shape[1]):
                res[i][j] = method(query[i:i + match.shape[0], j:j + match.shape[1]], match)
        if method in [self.cv_tm_sqdiff, self.cv_tm_sqdiff_normed]:
            top_left = np.argmin(res)
        else:
            top_left = np.argmax(res)
        return np.unravel_index(top_left, res.shape)

    # 平方差匹配
    def cv_tm_sqdiff(self, x, y):
        return np.sum(np.square(x - y))

    def cv_tm_sqdiff_normed(self, x, y):
        return np.std(np.square(x - y))

    def cv_tm_ccorr(self, x, y):
        return np.sum(x * y)

    def cv_tm_ccorr_normed(self, x, y):
        return np.std(x * y)

    def cv_tm_ccoeff(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        x -= np.mean(x)
        y -= np.mean(y)
        return np.sum(x * y)

    def cv_tm_ccoeff_normed(self, x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        x -= np.mean(x)
        y -= np.mean(y)
        return np.std(x * y)


def cv_mt(query, match, method=None):
    if method is None:
        method = cv2.TM_SQDIFF
    res = cv2.matchTemplate(query, match, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        return min_loc
    else:
        return max_loc


if __name__ == '__main__':
    query = cv2.imread('img/lena512color.png')
    match = query[200:450, 150:350]
    # cv2.imwrite('img/detected.png', match)
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)  # change to gray image
    match_gray = cv2.cvtColor(match, cv2.COLOR_BGR2GRAY)

    # model = NpMt()
    #
    # x, y = model.detected(query_gray, match_gray, method=model.cv_tm_sqdiff)
    # print('mt_cv_tm_sqdiff', x, y)
    # detected = query.copy()
    # cv2.rectangle(detected, (y, x), (y+match.shape[1], x+match.shape[0]), (50, 50, 180), 2)
    # cv2.imwrite('img/mt_cv_tm_sqdiff.png', detected)
    #
    # x, y = model.detected(query_gray, match_gray, method=model.cv_tm_sqdiff_normed)
    # print('cv_tm_sqdiff_normed', x, y)
    # detected = query.copy()
    # cv2.rectangle(detected, (y, x), (y+match.shape[1], x+match.shape[0]), (50, 50, 180), 2)
    # cv2.imwrite('img/mt_cv_tm_sqdiff_normed.png', detected)
    #
    # x, y = model.detected(query_gray, match_gray, method=model.cv_tm_ccoeff)
    # print('cv_tm_ccoeff', x, y)
    # detected = query.copy()
    # cv2.rectangle(detected, (y, x), (y+match.shape[1], x+match.shape[0]), (50, 50, 180), 2)
    # cv2.imwrite('img/mt_cv_tm_ccoeff.png', detected)
    #
    # x, y = model.detected(query_gray, match_gray, method=model.cv_tm_ccoeff_normed)
    # print('cv_tm_ccoeff_normed', x, y)
    # detected = query.copy()
    # cv2.rectangle(detected, (y, x), (y+match.shape[1], x+match.shape[0]), (50, 50, 180), 2)
    # cv2.imwrite('img/mt_cv_tm_ccoeff_normed.png', detected)
    #
    # x, y = model.detected(query_gray, match_gray, method=model.cv_tm_ccorr)
    # print('cv_tm_ccorr', x, y)
    # detected = query.copy()
    # cv2.rectangle(detected, (y, x), (y+match.shape[1], x+match.shape[0]), (50, 50, 180), 2)
    # cv2.imwrite('img/mt_cv_tm_ccorr.png', detected)
    #
    # x, y = model.detected(query_gray, match_gray, method=model.cv_tm_ccorr_normed)
    # print('cv_tm_ccorr_normed', x, y)
    # detected = query.copy()
    # cv2.rectangle(detected, (y, x), (y+match.shape[1], x+match.shape[0]), (50, 50, 180), 2)
    # cv2.imwrite('img/mt_cv_tm_ccorr_normed.png', detected)
    print(query_gray)
    cv_mt(query_gray, match_gray)
