import os
import pandas as pd
from PIL import Image


def resize_image(path_open, width, height, path_save, train=True):
    """ Function resizing images to desired values.
        path_open - path to directory with images
        width, height - size of output image
        path_save - directory where resized images should be saved
        train - boolean parameter, test data doest have bounding boxes
        Bounding boxes are scaled too."""
    if train:
        frame = pd.read_csv(r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\labels.csv')
        frame_resized = frame.copy()
        x_columns = [column for column in frame_resized.columns if 'X' in column]
        y_columns = [column for column in frame_resized.columns if 'Y' in column]
        x_ratios = []
        y_ratios = []
    files = os.listdir(path_open)
    alias = 'resized_'
    for file in files:
        img = Image.open(os.path.join(path_open, file))
        y_squeeze_ratio = img.height / height
        x_squeeze_ratio = img.width / width
        if train:
            x_ratios.append(x_squeeze_ratio)
            y_ratios.append(y_squeeze_ratio)
            frame_resized.loc[frame_resized.name == f'"Train/pos/{file}"', x_columns] = \
                frame_resized.loc[frame_resized.name == f'"Train/pos/{file}"', x_columns].apply(
                    lambda x: x / x_squeeze_ratio, axis=1)
            frame_resized.loc[frame_resized.name == f'"Train/pos/{file}"', y_columns] = \
                frame_resized.loc[frame_resized.name == f'"Train/pos/{file}"', y_columns].apply(
                    lambda y: y / y_squeeze_ratio, axis=1)
            frame_resized.loc[frame_resized.name == f'"Train/pos/{file}"', 'name'] = file
        img = img.resize((width, height), Image.ANTIALIAS)
        img.save(os.path.join(path_save, f'{alias}{file}'))
    if train:
        frame_resized['x_squeeze'] = x_ratios
        frame_resized['y_squeeze'] = y_ratios
        frame_resized.to_csv('resized_frame.csv', index=False)

