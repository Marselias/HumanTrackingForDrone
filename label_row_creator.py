import numpy as np
import os


from unpack_annotation import unpack_annotation


def create_row(annotation):
    """Function for creating single row of labels from annotations """

    path = os.path.join('annotations', annotation)
    name, number, centers, boxes = unpack_annotation(path)

    # max number is 8, 16 points for centers and 32 points for boxes, number, name
    name = np.array([name])
    buf = np.zeros(49, dtype=np.int32)
    buf[0] = number

    # adding centers to labels
    for i, center in enumerate(centers):
        buf[1 + i*2] = center[0]
        buf[1 + i*2 + 1] = center[1]

    # adding boxes to label
    for i, box in enumerate(boxes):
        buf[17 + i*4] = box[0]
        buf[17 + i * 4 + 1] = box[1]
        buf[17 + i * 4 + 2] = box[2]
        buf[17 + i * 4 + 3] = box[3]

    return np.hstack((name, buf))
