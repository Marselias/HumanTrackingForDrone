import cv2
import numpy as np
import os


def noise_image(path_open, path_save, noise_level=40):
    """ Function noising image.
        path_open - path to directory with images
        path_save - directory where resized images should be saved
        noise_level - level of noise
        Bounding boxes are not affected."""

    alias = 'noised_'
    files = os.listdir(path_open)
    for file in files:
        core = '_'.join(file.split('_')[1:])
        img = cv2.imread(os.path.join(path_open, file))
        img = np.int32(img)
        noise = np.random.randint(-noise_level, noise_level, size=img.shape, dtype=np.int32)
        img = np.clip(img + noise, 0, 255)
        img = np.uint8(img)
        cv2.imwrite(os.path.join(path_save, f'{alias}{core}'), img)


noise_image(r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Resized',
            r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Noised')
