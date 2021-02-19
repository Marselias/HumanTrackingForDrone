import os
import pandas as pd
from PIL import Image, ImageOps


def flip_image(path_open, path_save, path_frame):

    frame = pd.read_csv(path_frame)
    frame_flip = frame.copy()
    files = os.listdir(path_open)
    alias = 'flipped_'

    for file in files:
        core = '_'.join(file.split('_')[1:])
        img = Image.open(os.path.join(path_open, file))
        height = img.height
        img = ImageOps.flip(img)
        img.save(os.path.join(path_save, f'{alias}{core}'))
        slice_ = frame_flip[frame_flip.name == core]
        y_columns = [column for column in frame_flip.columns if "Y" in column and slice_[column].item() > 0]
        frame_flip.loc[frame_flip.name == core, y_columns] = frame_flip.loc[frame_flip.name == core, y_columns].apply(
            lambda y: height - y, axis=1)
        frame_flip.to_csv('flipped_frame.csv', index=False)


flip_image(r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Resized',
           r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Flipped',
           r'C:\Users\jangl\PycharmProjects\HumanTracking\DataAugmentation\resized_frame.csv')