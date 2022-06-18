import os
import cv2
import csv
import torch


def find_car(input_dir, output_cars ='output.csv'):
    input_dir = 'data'
    cars, imgs = ['car', 'truck', 'bus'], []

    for file_name in os.listdir(input_dir):
        imgs.append(cv2.imread(os.path.join(input_dir, file_name)))

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(imgs)

    with open(output_cars, 'w', newline='') as f:
        writer = csv.writer(f)

        for i, file_name in enumerate(os.listdir(input_dir)):
            res = [n in results.pandas().xyxy[i]['name'].unique() for n in cars]
            writer.writerow([file_name, bool(sum(res))])

if __name__ == '__main__':
    input_dir = 'data'

    find_car('data')
