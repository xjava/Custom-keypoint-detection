import json
import argparse
import os
from PIL import Image
from common_dataset_util import read_json, write_json, filter_label_studio_datasets
import cv2


def draw_image(input_image_path, output_image_path, keypoints):
    image = cv2.imread(input_image_path)
    color = (255, 0, 0)  # Line color in BGR (blue, green, red)
    thickness = 3  # Line thickness
    if image is None:
        raise ValueError("Image not found or the path is incorrect")

    x1, y1, v1 = round(keypoints[0]), round(keypoints[1]), keypoints[2]
    x2, y2, v2 = round(keypoints[3]), round(keypoints[4]), keypoints[5]
    x3, y3, v3 = round(keypoints[6]), round(keypoints[7]), keypoints[8]
    x4, y4, v4 = round(keypoints[9]), round(keypoints[10]), keypoints[11]

    # Draw the line
    cv2.line(image, (x1, y1),  (x2, y2), (255, 0, 0), thickness)
    cv2.line(image, (x2, y2), (x3, y3), (0, 255, 0), thickness)
    cv2.line(image, (x3, y3), (x4, y4), (0, 0, 255), thickness)
    cv2.line(image, (x4, y4), (x1, y1), (0, 255, 255), thickness)

    # Save the image with the drawn line
    cv2.imwrite(output_image_path, image)

def resize_dataset_in_directory(input_dir, output_dir):

    input_json_path = os.path.join(input_dir, 'all.json')
    input_dir_images = os.path.join(input_dir, 'images')

    output_dir_images = os.path.join(output_dir, 'images')

    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)

    # Read the JSON data from the input file
    data = read_json(input_json_path)
    print("Total read from Label Studio JSON file: {}".format(len(data)))
    annotations = data['annotations']
    images = data['images']
    for annotation in annotations:
        image_id = annotation['image_id']
        image = None
        for im in images:
            if im['id'] == image_id:
                image = im
                break
        assert image is not None
        filename = image['file_name']

        assert image['width'] == annotation['width'] and image['height'] == annotation['height']
        keypoints = annotation['keypoints']

        input_image_path = os.path.join(input_dir_images, filename)
        output_image_path = os.path.join(output_dir_images, filename)

        draw_image(input_image_path, output_image_path, keypoints)
        print(f"Draw and saved to {output_image_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resize all images in a directory while maintaining their aspect ratios.')
    parser.add_argument('--input_dir', type=str, default='/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset',
                        help='Path to the input directory containing images. Default is "input_images".')
    parser.add_argument('--output_dir', type=str, default='/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/test_draw',
                        help='Path to the output directory to save resized images. Default is "output_images".')
    args = parser.parse_args()

    resize_dataset_in_directory(args.input_dir, args.output_dir)
    print(f"All images resized and saved to {args.output_dir}")
