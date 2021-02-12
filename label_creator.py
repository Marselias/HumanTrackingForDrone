import os
import pandas as pd


from label_row_creator import create_row


def create_frame(save=False, save_path=None):
    columns = ['name', 'obj_number', 'c1X', 'c1Y', 'c2X', 'c2Y', 'c3X', 'c3Y', 'c4X', 'c4Y', 'c5X', 'c5Y', 'c6X', 'c6Y',
           'c7X', 'c7Y', 'c8X1', 'c8Y2', 'b1Xmin', 'b1Ymin', 'b1Xmax', 'b1Ymax', 'b2Xmin', 'b2Ymin', 'b2Xmax', 'b2Ymax',
           'b3Xmin', 'b3Ymin', 'b3Xmax', 'b3Ymax', 'b4Xmin', 'b4Ymin', 'b4Xmax', 'b4Ymax', 'b5Xmin', 'b5Ymin', 'b5Xmax',
           'b5Ymax', 'b6Xmin', 'b6Ymin', 'b6Xmax', 'b6Ymax', 'b7Xmin', 'b7Ymin', 'b7Xmax', 'b7Ymax', 'b8Xmin', 'b8Ymin',
           'b8Xmax', 'b8Ymax']
    frame = pd.DataFrame(columns=columns)

    for i, path in enumerate(os.listdir('annotations')):
        frame.loc[i] = create_row(path)
    if save:
        frame.to_csv(save_path, index=False)
    print(frame)


create_frame(save=True, save_path='labels.csv')
