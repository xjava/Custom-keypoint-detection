import json
import argparse
import os
from PIL import Image
from common_dataset_util import read_json, write_json, filter_label_studio_datasets
import cv2


def resize_image(input_image_path, output_image_path, max_size, scale_down_only):
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError("Image not found or the path is incorrect")

    # Get original dimensions
    (h, w) = image.shape[:2]

    if scale_down_only and max(h, w) < max_size:
        ratio = 1.0
    else:
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
    old_dimensions = (w, h)
    return old_dimensions, new_dimensions


def resize_dataset_in_directory(input_dir, output_dir, max_size, scale_down_only):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all files in the directory
    files = os.listdir(input_dir)
    files.sort()  # Optional: sort the files if needed

    # Loop through all files and rename them
    valid_extensions = {".jpg"}

    for filename in files:
        # Check if the file has a valid extension
        if os.path.splitext(filename)[1].lower() in valid_extensions:
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)
            (old_width, old_height), (new_width, new_height) = resize_image(input_image_path, output_image_path, max_size, scale_down_only)
            if old_width > new_width:
                print(
                    f"Decrease sized {input_image_path} {old_width, old_height} and saved to {output_image_path} {new_width, new_height}")
            else:
                print(
                    f"Keep original size of  {input_image_path} {old_width, old_height} and saved to {output_image_path} {new_width, new_height}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resize all images in a directory while maintaining their aspect ratios.')
    parser.add_argument('--input_dir', type=str, default='/Users/nikornlansa/Downloads/ray',
                        help='Path to the input directory containing images. Default is "input_images".')
    parser.add_argument('--output_dir', type=str, default='/Users/nikornlansa/Downloads/ray_resize',
                        help='Path to the output directory to save resized images. Default is "output_images".')
    parser.add_argument('--max_size', type=int, default=1536,
                        help='Maximum size of the resized images. Default is 1536.')
    parser.add_argument('--scale_down_only', type=bool, default=True,
                        help='Don\'t increase the size if it is smaller than the maximum size.')


    args = parser.parse_args()

    resize_dataset_in_directory(args.input_dir, args.output_dir, args.max_size, args.scale_down_only)
    print(f"All images resized and saved to {args.output_dir}")
