import cv2
import numpy as np
import os


def change_intensity(path_open, path_save, change_level=80):
    """ Function changing intensity of a image.
        path_open - path to directory with images
        path_save - directory where resized images should be saved
        change_level - level of intensity change
        Bounding boxes are not affected."""

    alias = 'intensity_'
    files = os.listdir(path_open)
    for file in files:
        core = '_'.join(file.split('_')[1:])
        img = cv2.imread(os.path.join(path_open, file))
        img = np.int32(img)
        change = np.ones(shape=img.shape, dtype=np.int32)*change_level
        img_up = np.clip(img + change, 0, 255)
        img_up = np.uint8(img_up)
        img_down = np.clip(img - change, 0, 255)
        img_down = np.uint8(img_down)
        cv2.imwrite(os.path.join(path_save, f'{alias}up_{core}'), img_up)
        cv2.imwrite(os.path.join(path_save, f'{alias}down_{core}'), img_down)


change_intensity(r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Resized',
                 r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Intensity')
