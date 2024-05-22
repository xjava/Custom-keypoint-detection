import json
import argparse
import os
from PIL import Image
from common_dataset_util import read_json, write_json, filter_label_studio_datasets


def resize_image(input_image_path, output_image_path, max_size):
    with Image.open(input_image_path) as image:
        original_width, original_height = image.size
        max_width, max_height = max_size

        # Calculate the new dimensions while maintaining the aspect ratio
        aspect_ratio = original_width / original_height

        if original_width / max_width > original_height / max_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_image.save(output_image_path)
        return new_width, new_height


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
    parser.add_argument('--max_width', type=int, default=1024,
                        help='Maximum width of the resized images. Default is 800.')
    parser.add_argument('--max_height', type=int, default=1024,
                        help='Maximum height of the resized images. Default is 600.')

    args = parser.parse_args()

    max_size = (args.max_width, args.max_height)

    resize_dataset_in_directory(args.input_dir, args.output_dir, max_size)
    print(f"All images resized and saved to {args.output_dir}")
