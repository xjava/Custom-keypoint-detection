import json
import copy
import argparse
import os
import shutil
import cv2
from coco_dataset_util import read_coco


def get_image_info_by_id(coco_data, image_id):
    for image in coco_data['images']:
        if image['id'] == image_id:
            return image
    return None


def get_annotate_info_by_image_id(coco_data, image_id):
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id:
            return annotation
    return None


def validate_coco_files(coco_json_path):
    error_dict = {}
    image_count = 0
    for file_path in coco_json_path:
        print(f'Checking.. Dataset: {file_path}')
        error = []
        error_dict[file_path] = error
        parent = os.path.dirname(file_path)
        filename, ext = os.path.splitext(os.path.basename(file_path))
        images_folder = os.path.join(parent, filename)

        coco_data = read_coco(file_path)

        for image_json in coco_data['images']:
            image_id = image_json['id']
            file_name = image_json['file_name']
            print(f'Checking.. image_id: {image_id}  file_name: {file_name} ')
            annotation = get_annotate_info_by_image_id(coco_data, image_id)
            if annotation is None:
                error.append(f'image_id: {image_id}  file_name: {file_name} no annotation found')

            image = cv2.imread(os.path.join(images_folder, file_name))
            # Get original dimensions
            (height, width, c) = image.shape
            if c != 3:
                error.append(f'image_id: {image_id}  file_name: {file_name} invalid image channel')
            if height != image_json['height'] and width != image_json['width']:
                error.append(f'image_id: {image_id}  file_name: {file_name} invalid image size')
            image_count = image_count + 1

        # Merge categories and create a mapping from old category IDs to new ones
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            image_json = get_image_info_by_id(coco_data, image_id)
            file_name = image_json['file_name']
            image_width = image_json['width']
            image_height = image_json['height']

            if 'bbox' in annotation:
                x, y, width, height = annotation['bbox']
                x, y, width, height = int(x), int(y), int(width), int(height)

                if x < 0 or x + width > image_width or y < 0 or y + height > image_height:
                    error.append(
                        f'image_id {image_id} , {file_name} :  invalid bbox size {x, y, width, height} image size {image_width, image_height}')
                # if image_id == 29 or image_id == 95:
                #     error.append(f'image_id {image_id} , {file_name} :  bbox size {x, y, width, height} image size {image_width, image_height}')

            if 'keypoints' in annotation:
                keypoints = annotation['keypoints']
                if keypoints is None or len(keypoints) != 12:
                    # raise Exception(f"Invalid keypoint found: {file_path}   image_id: {annotation['image_id']}")
                    error.append(f'image_id {image_id} , {file_name} : invalid keypoints length')

                for i in range(0, len(keypoints), 3):
                    kp_x = keypoints[i]
                    kp_y = keypoints[i + 1]
                    invalid_keypoints = False
                    if kp_x < 0 or kp_x > image_width or kp_y < 0 or kp_y > image_height:
                        invalid_keypoints = True

                if invalid_keypoints:
                    error.append(
                        f'image_id {image_id}, {file_name} image size {image_width, image_height} : invalid keypoints range {keypoints} ')

            else:
                # raise Exception(f"Invalid keypoint found: {file_path}   image_id: {annotation['image_id']}")
                error.append(f'image_id {image_id} , {file_name} :  keypoints not found')

    return image_count, error_dict


def validate_coco(coco_files):
    image_count, error_dict = validate_coco_files(coco_files)
    file_paths = error_dict.keys()
    for file_path in file_paths:
        errors = error_dict[file_path]
        if len(errors) == 0:
            print(f'{file_path} : No errors found.')
        else:
            print(file_path)
            for error in errors:
                print(error)
    print(f'Total image: {image_count}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=" ")
    parser.add_argument('-i', '--input', nargs='+', help="List of input COCO JSON files", required=False,
                        default=['/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/DocumentDetection.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/BoardDetection.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/CardDetection.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/DocumentDetection2.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/BookDetection.json'])

    args = parser.parse_args()

    validate_coco(args.input)
