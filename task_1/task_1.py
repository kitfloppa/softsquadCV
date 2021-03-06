import os
import cv2


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
