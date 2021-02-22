import cv2
import numpy as np
import os


def segmentation(path_open, path_save, k=15, iterations=100, epsilon=0.2):
    files = os.listdir(path_open)
    alias = 'segmented_'
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)

    for file in files:
        core = '_'.join(file.split('_')[1:])
        img = cv2.imread(os.path.join(path_open, file))
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.flatten()
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(img.shape)
        cv2.imwrite(os.path.join(path_save, f'{alias}{core}'), segmented_image)


segmentation(r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Resized',
             r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Segmented')
