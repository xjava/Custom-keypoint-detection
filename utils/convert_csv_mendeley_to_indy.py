import os
import csv
import argparse
import cv2
from common_dataset_util import write_json

def list_files(directory, extensions):
    # List all files in the given directory
    files = os.listdir(directory)
    # Filter out the files that end with .csv

    filtered_files = [file for file in files if file.endswith(extensions)]
    return filtered_files


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


# for convert a dataset from https://data.mendeley.com/datasets/x3nm4cxr83/3
def convert_to_indy_dataset(input_dir, output_dir, max_size, scale_down_only):
    jpg_files = list_files(input_dir, '.jpg')

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for jpg_file in jpg_files:
        # Get the filename without extension
        filename, ext = os.path.splitext(jpg_file)

        input_image_path = os.path.join(input_dir, filename + ".jpg")
        input_csv_path = os.path.join(input_dir, filename + ".csv")
        output_image_path = os.path.join(output_dir, filename + ".jpg")
        output_json_path = os.path.join(output_dir, filename + ".json")
        if not os.path.exists(input_csv_path):
            print(f'{filename}.csv not found. skipping...')
            break

        (old_width, old_height), (new_width, new_height) = resize_image(input_image_path, output_image_path, max_size,
                                                                        scale_down_only)

        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            row = next(reader)  # gets the first line
            top_left_x = float(row[0]) / old_width
            top_left_y = float(row[1]) / old_height
            bottom_left_x = float(row[2]) / old_width
            bottom_left_y = float(row[3]) / old_height
            bottom_right_x = float(row[4]) / old_width
            bottom_right_y = float(row[5]) / old_height
            top_right_x = float(row[6]) / old_width
            top_right_y = float(row[7]) / old_height
            annotations = []
            annotation = {
                'category_id': 1,
                'keypoints': [top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y,
                             bottom_left_x, bottom_left_y]
            }

            annotations.append(annotation)
            indy_json = {'annotations': annotations}
            write_json(output_json_path, indy_json)

        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy CSV files from input directory to output directory.')
    parser.add_argument('--input_dir', type=str, default='/Users/nikornlansa/Downloads/csv_datasets',
                        help='Path to the input directory')
    parser.add_argument('--output_dir', type=str, default='/Users/nikornlansa/Downloads/indy_dataset',
                        help='Path to the output directory')
    parser.add_argument('--max_size', type=int, default=1536,
                        help='Maximum size of the resized images. Default is 1536.')
    parser.add_argument('--scale_down_only', type=bool, default=True,
                        help='Don\'t increase the size if it is smaller than the maximum size.')
    args = parser.parse_args()

    convert_to_indy_dataset(args.input_dir, args.output_dir, args.max_size, args.scale_down_only)
