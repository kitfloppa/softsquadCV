import os
import cv2
import numpy as np

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
                cv2.imwrite(output_dir + f'/{file_name[:6]}rgb.jpg', merge_image)
    
if __name__ == '__main__':
    image_count = open('image_counter.txt').read()
    input_dir, output_dir = 'data', 'rgb_data'

    merge_channels(input_dir, output_dir)
    