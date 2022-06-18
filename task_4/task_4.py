import os
import cv2
import csv
import torch
import numpy as np
import pandas as pd


def get_color_name(file, color):
    minimum = 10000
    
    for i in range(len(file)):
        d = abs(color[0] - int(file.loc[i, "R"])) + abs(color[1] - int(file.loc[i, "G"])) + abs(color[2] - int(file.loc[i, "B"]))
        
        if(d <= minimum):
            minimum = d
            cname = file.loc[i, "color_name"]
    
    return cname


def find_color(input_dir, output_file='output_color.csv'):
    index=["color_name", "R", "G", "B"]
    color_csv = pd.read_csv('colors_name.csv', names=index, header=None)

    cars = ['car', 'truck', 'bus']
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
                    max, loc = 0, 0

                    for j in range(len(results.pandas().xyxy[i])):
                        target = results.pandas().xyxy[i].iloc[j]
                        
                        if max < int(target['xmax'] - target['xmin']) + int(target['ymax'] - target['ymin']):
                            max = int(target['xmax'] - target['xmin']) + int(target['ymax'] - target['ymin'])                             
                            loc = j

                    target = results.pandas().xyxy[i].iloc[loc]

                    x, y, w, h = int(target['xmin']), int(target['ymax']), int(target['xmax'] - target['xmin']), int(target['ymax'] - target['ymin'])
                    x, y, w, h = x + h // 5, y - h // 4, w // 2, h // 2
                    
                    img_bar, color = calc_metric(img, x, y, w, h)
            
                    color = get_color_name(color_csv, color)
                    writer.writerow([file_names[i], color])


def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])

    return bar, (red, green, blue)


def calc_metric(image, x, y, w, h):
    image = image[(y - h):(y-h//2), x:(x + w)]
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    percentages = (np.unique(label, return_counts=True)[1]) / data.shape[0]

    bar, color = create_bar(400, 200, center[np.argmax(percentages)])
    img_bar = np.hstack((bar,))

    return img_bar, color


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
    input_dir, output_dir = 'data', 'rgb_data'
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        merge_channels(input_dir, output_dir)

    find_color(output_dir)
