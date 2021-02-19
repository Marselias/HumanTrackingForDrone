import os
import pandas as pd
from PIL import Image, ImageOps


def flip_image(path_open, path_save, path_frame):

    frame = pd.read_csv(path_frame)
    frame_mirror = frame.copy()
    files = os.listdir(path_open)
    alias = 'mirrored_'

    for file in files:
        core = '_'.join(file.split('_')[1:])
        img = Image.open(os.path.join(path_open, file))
        width = img.width
        img = ImageOps.mirror(img)
        # img.save(os.path.join(path_save, f'{alias}{core}'))
        slice_ = frame_mirror[frame_mirror.name == core]
        x_columns = [column for column in frame_mirror.columns if "X" in column and slice_[column].item() > 0]
        frame_mirror.loc[frame_mirror.name == core, x_columns] = frame_mirror.loc[frame_mirror.name == core, x_columns].apply(
            lambda x: width - x, axis=1)
        frame_mirror.to_csv('mirrored_frame.csv', index=False)


flip_image(r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Resized',
           r'C:\Users\jangl\PycharmProjects\HumanTracking\Data\Train\Mirrored',
           r'C:\Users\jangl\PycharmProjects\HumanTracking\DataAugmentation\resized_frame.csv')