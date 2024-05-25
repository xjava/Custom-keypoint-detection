import json
import argparse
import os
from PIL import Image
from common_dataset_util import read_json, write_json, filter_label_studio_datasets
import cv2


def resize_image(input_image_path, output_image_path, max_size):
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError("Image not found or the path is incorrect")

    # Get original dimensions
    (h, w) = image.shape[:2]

    # Determine the scaling factor
    if w > h:
        ratio = max_size / float(w)
    else:
        ratio = max_size / float(h)

    # Calculate new dimensions
    new_dimensions = (int(w * ratio), int(h * ratio))

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    # Save the resized image to the output path
    cv2.imwrite(output_image_path, resized_image)
    return new_dimensions


def resize_dataset_in_directory(input_dir, output_dir, max_size):

    input_json_path = os.path.join(input_dir, 'label_studio.json')
    input_dir_images = os.path.join(input_dir, 'images')

    output_json_path = os.path.join(output_dir, 'label_studio.json')
    output_dir_images = os.path.join(output_dir, 'images')

    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)

    # Read the JSON data from the input file
    data = read_json(input_json_path)
    print("Total read from Label Studio JSON file: {}".format(len(data)))
    data = filter_label_studio_datasets(data)
    print("Total filtered from  Label Studio JSON file: {}".format(len(data)))
    for row in data:
        results = row['annotations'][0]['result']
        corners = None
        width = None
        height = None
        for result in results:
            if result['from_name'] == 'corners':
                corners = result
        if corners is not None:
            width = corners['original_width']
            height = corners['original_height']

        file_upload = row['file_upload']
        filename = file_upload[9:]  # 7ae5c878-IMG_2688.JPG ->IMG_2688.JPG
        input_image_path = os.path.join(input_dir_images, filename)
        output_image_path = os.path.join(output_dir_images, filename)
        new_width, new_height = resize_image(input_image_path, output_image_path, max_size)
        print(f"Resized {input_image_path} {width, height} and saved to {output_image_path} {new_width, new_height}")
        corners['original_width'] = new_width
        corners['original_height'] = new_height

    write_json(output_json_path, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resize all images in a directory while maintaining their aspect ratios.')
    parser.add_argument('--input_dir', type=str, default='input_dir',
                        help='Path to the input directory containing images. Default is "input_images".')
    parser.add_argument('--output_dir', type=str, default='output_dir',
                        help='Path to the output directory to save resized images. Default is "output_images".')
    parser.add_argument('--max_size', type=int, default=1024,
                        help='Maximum size of the resized images. Default is 1024.')

    args = parser.parse_args()

    resize_dataset_in_directory(args.input_dir, args.output_dir, args.max_size)
    print(f"All images resized and saved to {args.output_dir}")
