import os
import cv2
import csv
import torch
import numpy as np
import pandas as pd


def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    
    return bar, (red, green, blue)


def calc_metric(image, x, y, w, h):
    image = image[(y - h):y, x:(x + w)]
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    percentages = (np.unique(label, return_counts=True)[1]) / data.shape[0]

    bar, color = create_bar(400, 200, center[np.argmax(percentages)])
    img_bar = np.hstack((bar,))

    cv2.imshow('img', image)
    cv2.imshow('bar', img_bar)
    cv2.waitKey(0)

    return img_bar, color


def find_car(input_dir, output_cars ='output.csv'):
    os.chdir(input_dir)
    input_dir, output_dir = 'data', 'rgb_data'
    cars = ['car', 'truck', 'bus']
    imgs = []

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        merge_channels(input_dir, output_dir)

    for file_name in os.listdir(output_dir):
        imgs.append(cv2.imread(os.path.join(output_dir, file_name)))

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(imgs)

    if not os.path.exists(output_cars):
        with open(output_cars, 'w', newline='') as f:
            writer = csv.writer(f)

            for i, file_name in enumerate(os.listdir(output_dir)):
                res = [n in results.pandas().xyxy[i]['name'].unique() for n in cars]

                writer.writerow([file_name, bool(sum(res))])


def merge_channels(input_dir, output_dir):
    data = os.listdir(input_dir)
    channels, count,  = [], 0

    for i, file_name in enumerate(data):
        if count != 3:
            count += 1
            channels.append(cv2.imread(os.path.join(input_dir, file_name), 0))
            if count == 3:
                count = 0
                merge_image = cv2.merge([channels[0], channels[1], channels[2]])
                channels = []
                cv2.imwrite(output_dir + f'/{file_name[:5]}.jpg', merge_image)
    
if __name__ == '__main__':

    

    img = cv2.imread('input_dir_color/rgb_data/00020.jpg')

    img = [img]

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(img)

    cars = ['car', 'truck', 'bus']

    for i, img in enumerate(img):
        res = [n in results.pandas().xyxy[i]['name'].unique() for n in cars]

        if bool(sum(res)):
            max, loc = 0, 0

            for j in range(len(results.pandas().xyxy[i])):
                target = results.pandas().xyxy[i].iloc[j]
                if max < int(target['xmax'] - target['xmin']) + int(target['ymax'] - target['ymin']):
                    max = int(target['xmax'] - target['xmin']) + int(target['ymax'] - target['ymin'])
                    loc = j

            target = results.pandas().xyxy[i].iloc[loc]

            x, y, w, h = int(target['xmin']), int(target['ymax']), int(target['xmax'] - target['xmin']), int(target['ymax'] - target['ymin'])
            img_bar, color = calc_metric(img, x + h // 5, y - h // 4, w // 2, h // 2)

