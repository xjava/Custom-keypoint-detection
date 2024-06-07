import csv
import argparse
import os
from PIL import Image
from common_dataset_util import list_files
import cv2


def draw_image(input_image_path, output_image_path, csv_row):
    image = cv2.imread(input_image_path)
    (h, w) = image.shape[:2]
    thickness = 3  # Line thickness
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    csv_row = [float(i) for i in csv_row]

    x1, y1 = round(csv_row[0] * w), round(csv_row[1] * h)
    x2, y2 = round(csv_row[2] * w), round(csv_row[3] * h)
    x3, y3 = round(csv_row[4] * w), round(csv_row[5] * h)
    x4, y4 = round(csv_row[6] * w), round(csv_row[7] * h)

    # Draw the line
    cv2.line(image, (x1, y1),  (x2, y2), (255, 0, 0), thickness)
    cv2.line(image, (x2, y2), (x3, y3), (0, 255, 0), thickness)
    cv2.line(image, (x3, y3), (x4, y4), (0, 0, 255), thickness)
    cv2.line(image, (x4, y4), (x1, y1), (0, 255, 255), thickness)

    # Save the image with the drawn line
    cv2.imwrite(output_image_path, image)

def draw_dataset_in_directory(input_dir, output_dir):
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
        if not os.path.exists(input_csv_path):
            print(f'{filename}.csv not found. skipping...')
            break

        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            row = next(reader)  # gets the first line
            draw_image(input_image_path, output_image_path, row)

        print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resize all images in a directory while maintaining their aspect ratios.')
    parser.add_argument('--input_dir', type=str, default='/Users/nikornlansa/Downloads/temp_csv_datasets',
                        help='Path to the input directory containing images. Default is "input_images".')
    parser.add_argument('--output_dir', type=str, default='/Users/nikornlansa/Downloads/draw_csv_datasets',
                        help='Path to the output directory to save resized images. Default is "output_images".')
    args = parser.parse_args()

    draw_dataset_in_directory(args.input_dir, args.output_dir)
    print(f"All images resized and saved to {args.output_dir}")
