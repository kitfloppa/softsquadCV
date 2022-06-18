import os
import cv2
import csv
import torch
import numpy as np
import pandas as pd


def check_stop_signals(input_dir, output_file="output_lights.csv"):
    cars = ['car', 'truck', 'bus']
    state = ['nolights', 'lights']
    imgs, file_names = [], []

    for file_name in os.listdir(input_dir):
        imgs.append(cv2.imread(os.path.join(input_dir, file_name)))
        file_names.append(file_name)
        
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(imgs)

    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            for i, img in enumerate(imgs):
                res = [n in results.pandas().xyxy[i]['name'].unique() for n in cars]

                if bool(sum(res)):
                    model_lights = torch.hub.load('../yolov5', 'custom', path='lights.pt', source='local')
                    results_light = model_lights(img)
                    
                    print(results_light)

                    res = [n in results_light.pandas().xyxy[0]['name'].unique() for n in state]

                    light_state = 'Off'

                    if bool(sum(res)): light_state = 'On'
            
                    writer.writerow([file_names[i], light_state])
                else:
                    writer.writerow([file_names[i], 'no_cars'])


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
    torch.cuda.empty_cache()

    input_dir, output_dir = 'data', 'rgb_data'
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        merge_channels(input_dir, output_dir)

    check_stop_signals(output_dir)
