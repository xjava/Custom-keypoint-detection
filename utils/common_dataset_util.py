import json


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data


def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def filter_label_studio_datasets(json_data):
    filtered_json = []
    errors = []
    max_xy = 100.0
    for row in json_data:
        label_id = row['id']
        results = row['annotations'][0]['result']

        corners = None
        used_for_training = None
        for result in results:

            if result['from_name'] == 'corners':
                corners = result
            if result['from_name'] == 'used_for_training':
                used_for_training = result

        # just want to validate all labeled corners
        if corners is not None:
            points = corners['value']['points']
            p1x = points[0][0]
            p1y = points[0][1]
            p2x = points[1][0]
            p2y = points[1][1]
            p3x = points[2][0]
            p3y = points[2][1]
            p4x = points[3][0]
            p4y = points[3][1]

            if len(points) != 4:
                errors.append("incorrect corner length id = {}".format(label_id))
            elif p1x > max_xy or p1x > max_xy or p1y > max_xy or p2x > max_xy or p2y > max_xy or p3x > max_xy or p3y > max_xy or p4x > max_xy or p4y > max_xy:
                errors.append("incorrect size id = {}".format(label_id))
            elif p1x > p2x or p1x > p3x or p4x > p2x or p4x > p3x:
                errors.append("incorrect coordinate x id = {}".format(label_id))
            elif p1y > p4y or p1y > p3y or p2y > p4y or p2y > p3y:
                errors.append("incorrect coordinate y id = {}".format(label_id))

        if used_for_training is not None:
            if used_for_training['value']['choices'][0] == 'ใช้':
                filtered_json.append(row)

    if len(errors) > 0:
        for error in errors:
            print(error)
        raise TypeError("Total corner coordinate values are incorrect in {} files.".format(len(errors)))

    return filtered_json
