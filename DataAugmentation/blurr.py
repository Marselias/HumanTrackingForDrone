import cv2
import numpy as np
import os


def blur_image(path_open, path_save, kernel_size=5):
    """ Function flipping image.
        path_open - path to directory with images
        path_save - directory where resized images should be saved
        Bounding boxes are not affected."""

    alias = 'blurred_'
    files = os.listdir(path_open)
    kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.float32)/(kernel_size**2)

    for file in files:
        core = '_'.join(file.split('_')[1:])
        img = cv2.imread(os.path.join(path_open, file))
        img = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(os.path.join(path_save, f'{alias}{core}'), img)


blur_image(r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Resized',
           r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Blurred')
