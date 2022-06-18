import cv2
import numpy as np


def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    
    return bar, (red, green, blue)


def calc_metric(image, x, y, w, h):
    image = image[(y - h):y, x:(x + w)]
    cv2.imshow('Image', image)
    data = np.float32(image).reshape((-1, 3))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    percentages = (np.unique(label, return_counts=True)[1]) / data.shape[0]

    bar, color = create_bar(400, 200, center[np.argmax(percentages)])
    img_bar = np.hstack((bar,))

    return img_bar, color

if __name__ == '__main__':
    image_path = 'data/00071.jpg'

    img = cv2.imread(image_path)

    color_bar, color = calc_metric(img, 100, 600, 900, 250)

    cv2.imshow('Color bar', color_bar)
    cv2.waitKey(0)
