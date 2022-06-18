import os
import cv2
import csv
import torch
import numpy as np


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
    find_car('input_dir', 'output_cars.csv')
    