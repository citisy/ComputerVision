import numpy as np
import cv2
import random


def GaussianNoise(image, percetage=0.1, mu=0., sigma=1.):
    G_Noiseimg = image
    size = image.shape[0] * image.shape[1]

    noise_pos = [random.gauss(mu, sigma) for _ in range(size)]
    for i in range(size):
        x = i // image.shape[1]
        y = i % image.shape[1]
        G_Noiseimg[x][y] += np.uint8(255 * percetage * noise_pos[i] / 5)

    return G_Noiseimg


def SaltAndPepper(src, percetage=0.01):
    SP_NoiseImg = src
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randX = np.random.random_integers(0, src.shape[0] - 1)
        randY = np.random.random_integers(0, src.shape[1] - 1)
        if np.random.random_integers(0, 1) == 0:
            SP_NoiseImg[randX, randY] = 0
        else:
            SP_NoiseImg[randX, randY] = 255
    return SP_NoiseImg


if __name__ == '__main__':
    img = cv2.imread('train_img/query.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = GaussianNoise(img)
    # cv2.imwrite('img/gauss_noise.png', gauss)
    cv2.imshow('rgb', gauss)
    cv2.waitKeyEx()
    cv2.destroyAllWindows()

    salt = SaltAndPepper(img)
    # cv2.imwrite('img/salt_noise.png', salt)
    cv2.imshow('rgb', salt)
    cv2.waitKeyEx()
    cv2.destroyAllWindows()
