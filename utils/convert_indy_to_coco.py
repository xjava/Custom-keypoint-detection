import os
import argparse
import cv2
from common_dataset_util import write_json, read_json, get_polygon_bounding_box, get_polygon_area


def list_files(directory, extensions):
    # List all files in the given directory
    files = os.listdir(directory)
    # Filter out the files that end with .csv

    filtered_files = [file for file in files if file.endswith(extensions)]
    return filtered_files


# for convert a dataset from https://data.mendeley.com/datasets/x3nm4cxr83/3
def convert_indy_to_coco(input_dir):
    jpg_files = list_files(input_dir, '.JPG')
    jpg_files.sort()


    images = []
    annotations = []
    categories = [{
        "id": 1,
        "name": "document",
        "supercategory": "",
        "color": "#ffff32",
        "keypoint_colors": [
            "#ff0000",
            "#00ff00",
            "#ff00ff",
            "#0000ff"
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
                1,
                4
            ],
            [
                2,
                3
            ],
            [
                3,
                4
            ]
        ]}]
    coco_json = {'images': images, 'categories': categories, 'annotations': annotations}

    image_id = 1
    for jpg_file in jpg_files:
        print(os.path.join(input_dir, jpg_file))
        # Get the filename without extension
        filename, ext = os.path.splitext(jpg_file)

        indy_json_path = os.path.join(input_dir, filename + ".json")
        if not os.path.exists(indy_json_path):
            continue

        img = cv2.imread(os.path.join(input_dir, jpg_file))
        if img is None:
            raise ValueError("Image not found or the path is incorrect")

        # Get original dimensions
        (height, width) = img.shape[:2]

        indy_json = read_json(indy_json_path)
        category_id = indy_json['annotations'][0]['category_id']
        keypoints = indy_json['annotations'][0]['keypoints']
        top_left_x = round(float(keypoints[0]) * width)
        top_left_y = round(float(keypoints[1]) * height)
        top_right_x = round(float(keypoints[2]) * width)
        top_right_y = round(float(keypoints[3]) * height)
        bottom_right_x = round(float(keypoints[4]) * width)
        bottom_right_y = round(float(keypoints[5]) * height)
        bottom_left_x = round(float(keypoints[6]) * width)
        bottom_left_y = round(float(keypoints[7]) * height)
        x = [top_left_x, top_right_x, bottom_right_x, bottom_left_x]
        y = [top_left_y, top_right_y, bottom_right_y, bottom_left_y]

        image = {'id': image_id,
                 "file_name": jpg_file,
                 "category_ids": [category_id],
                 "annotated": True,
                 "annotating": [],
                 "num_annotations": 1,
                 "metadata": {},
                 }
        annotation = {'id': image_id,
                      "image_id": image_id,
                      "category_id": category_id,
                      'ignore': 0,
                      "iscrowd": 0
                      }
        images.append(image)
        annotations.append(annotation)

        image["width"] = width
        image["height"] = height
        annotation["width"] = width
        annotation["height"] = height

        annotation['bbox'] = get_polygon_bounding_box(x, y)
        annotation['keypoints'] = [top_left_x, top_left_y, 2, top_right_x, top_right_y, 2,
                                   bottom_right_x, bottom_right_y, 2, bottom_left_x, bottom_left_y,
                                   2]  # https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md Each keypoint has a 0-indexed location x,y and a visibility flag v defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.
        annotation['num_keypoints'] = 4
        annotation['segmentation'] = [top_left_x, top_left_y, top_right_x, top_right_y,
                                      bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y]
        annotation['area'] = get_polygon_area(x, y)

        image_id = image_id + 1

    return coco_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy CSV files from input directory to output directory.')
    parser.add_argument('--input_dir', type=str, default='dataset/coco/MendeleyDetection',
                        help='Path to the input directory')
    args = parser.parse_args()


    coco_name = os.path.basename(args.input_dir)
    parent = os.path.dirname(args.input_dir)
    os.path.abspath(os.path.join(args.input_dir, os.pardir))
    coco_output_path = os.path.join(parent, coco_name + ".json")

    data = convert_indy_to_coco(args.input_dir)
    write_json(coco_output_path, data)

    print(f"Data has been written to {coco_output_path}")
