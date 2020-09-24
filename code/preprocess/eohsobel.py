import numpy as np
from PIL import Image
from preprocess.HaarLikeFeature import FeatureType
from functools import partial
import os
import cv2


def ensemble_vote(int_img, classifiers):

    return 1 if sum([c.get_vote(int_img) for c in classifiers]) >= 0 else 0


def ensemble_vote_all(int_imgs, classifiers):

   # Classifies given list of integral images (numpy arrays) using classifiers,

    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, int_imgs))


def reconstruct(classifiers, img_size):

    #Creates an image by putting all given classifiers on top of each other


    image = np.zeros(img_size)
    for c in classifiers:
        # map polarity: -1 -> 0, 1 -> 1
        polarity = pow(1 + c.polarity, 2)/4
        if c.type == FeatureType.TWO_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if y >= c.height/2:
                        sign = (sign + 1) % 2
                    image[c.top_left[1] + y, c.top_left[0] + x] += 1 * sign * c.weight
        elif c.type == FeatureType.TWO_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x >= c.width/2:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x % c.width/3 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if x % c.height/3 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.FOUR:
            sign = polarity
            for x in range(c.width):
                if x % c.width/2 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    if x % c.height/2 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
    image -= image.min()
    image /= image.max()
    image *= 255
    result = Image.fromarray(image.astype(np.uint8))
    return result


def load_images(path):

    images = []
    for _file in os.listdir(path):
        if _file.endswith('.jpg'):
            img_arr = np.array(Image.open((os.path.join(path, _file))), dtype=np.float64)
            img_arr /= img_arr.max()
            images.append(img_arr)
    return sobel(images,path)

#sobel mask
def sobel(img,path):

    container = np.copy(img)

    size = container.shape


    for i in range(1, size[0] - 1):

        for j in range(1, 0, - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container


