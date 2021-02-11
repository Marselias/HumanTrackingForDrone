def unpack_annotation(path):
    """Unpacking labels from INFRIAPerson Dataset annotations
    Returns 3 vars: how many objects, centers and bounding boxes coordinates."""
    buffer = []
    with open(path, 'r') as file:
        lines = file.read()

    lines = lines.splitlines()
    for line in lines:
        if not line.startswith('#') and line:
            buffer.append(line)

    # How many person-like objects in photo
    how_many = 0
    for line in buffer:
        if 'Objects with ground truth' in line:
            how_many = int((line.replace(' ', '').split(':')[1][0]))
            break

    person_id = []
    for i in range(how_many):
        person_id.append(f'{i+1} "PASperson"')

    # Centers of objects
    centers = []
    which_one = 0
    for line in buffer:
        if which_one == how_many:
            break
        if person_id[which_one] + ' (X, Y)' in line:
            buf = line.replace(" ", "").split(':')[1]
            buf = buf.replace('(', "").replace(')', '').split(',')
            centers.append((int(buf[0]), int(buf[1])))
            which_one += 1

    # Bounding boxes of objects
    boxes = []
    which_one = 0
    for line in buffer:
        if which_one == how_many:
            break
        if person_id[which_one] + ' (Xmin, Ymin)' in line:
            buf = line.replace(" ", "").split(':')[1]
            buf = buf.replace('(', "").replace(')', '').split('-')
            buf0 = buf[0].split(',')
            buf1 = buf[1].split(',')
            boxes.append((int(buf0[0]), int(buf0[1]), int(buf1[0]), int(buf1[1])))
            which_one += 1

    return how_many, centers, boxes
