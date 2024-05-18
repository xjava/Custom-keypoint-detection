import json
import argparse
import numpy as np


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data


def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def get_polygon_area(x, y):
    """https://en.wikipedia.org/wiki/Shoelace_formula"""

    assert len(x) == len(y)

    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_polygon_bounding_box(x, y):
    assert len(x) == len(y)

    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
    return [x1, y1, x2 - x1, y2 - y1]


def filter_data(json_data):
    filter_json = []
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
                filter_json.append(row)

    if len(errors) > 0:
        for error in errors:
            print(error)
        raise TypeError("Total corner coordinate values are incorrect in {} files.".format(len(errors)))

    return filter_json


def convert_to_coco(json_data):
    categories = [{
        "id": 15,
        "name": "corner",
        "color": "#dae182",
        "keypoint_colors": [
            "#bf5c4d",
            "#d99100",
            "#4d8068",
            "#0d2b80"
        ],
        "keypoints": [
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left"
        ],
        "skeleton": [
            [
                1,
                2
            ],
            [
                2,
                3
            ],
            [
                3,
                4
            ],
            [
                4,
                1
            ]
        ]}]
    images = []
    annotations = []
    coco_json = {'images': images, 'categories': categories, 'annotations': annotations}
    for row in json_data:
        image_id = row['id']
        file_upload = row['file_upload']
        filename = file_upload[9:]  # 7ae5c878-IMG_2688.JPG ->IMG_2688.JPG
        results = row['annotations'][0]['result']

        image = {'id': image_id,
                 "file_name": filename,
                 "dataset_id": 6,
                 "category_ids": [15],
                 "annotated": True,
                 "annotating": [],
                 "num_annotations": 1,
                 "metadata": {},
                 }
        annotation = {'id': image_id,
                      "image_id": image_id,
                      "dataset_id": 6,
                      "category_id": 15,
                      'ignore': 0,
                      "iscrowd": 0
                      }

        images.append(image)
        annotations.append(annotation)

        corners = None
        for result in results:
            if result['from_name'] == 'corners':
                corners = result

        if corners is not None:
            width = corners['original_width']
            height = corners['original_height']
            points = corners['value']['points']

            # https://github.com/xjava/label-studio-converter/blob/master/label_studio_converter/converter.py
            points_abs = [
                (x / 100 * width, y / 100 * height) for x, y in points
            ]
            x, y = zip(*points_abs)

            image["width"] = width
            image["height"] = height

            annotation["width"] = width
            annotation["height"] = height
            annotation['bbox'] = get_polygon_bounding_box(x, y)
            annotation['keypoints'] = [points_abs[0][0], points_abs[0][1], 2, points_abs[1][0], points_abs[1][1], 2,
                                       points_abs[2][0], points_abs[2][1], 2, points_abs[3][0], points_abs[3][1],
                                       2]  # https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md Each keypoint has a 0-indexed location x,y and a visibility flag v defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.
            annotation['num_keypoints'] = 4
            annotation['segmentation'] = [
                [coord for point in points_abs for coord in point]
            ]
            annotation['area'] = get_polygon_area(x, y)

    return coco_json


def main():
    """
    Main function to parse input and output filenames from command-line arguments.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Read and process a JSON file.')

    # Add arguments for input and output filenames
    parser.add_argument('--input_file', type=str, default='dataset/annotations/label_studio.json',
                        help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, default='dataset/annotations/coco_data.json',
                        help='Path to the output file where processed data will be written')

    # Parse the arguments
    args = parser.parse_args()

    # Read the JSON data from the input file
    data = read_json(args.input_file)
    print("Total read from JSON file: {}".format(len(data)))
    data = filter_data(data)
    print("Total filter from JSON file: {}".format(len(data)))
    data = convert_to_coco(data)
    write_json(args.output_file, data)

    print(f"Data has been written to {args.output_file}")


if __name__ == '__main__':
    main()
