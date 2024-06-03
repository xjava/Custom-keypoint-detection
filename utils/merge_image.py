import os
import shutil
import argparse


def copy_jpg_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    seen_files = set()

    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                if file in seen_files:
                    raise Exception(f"Duplicate file found: {file}")
                seen_files.add(file)
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_folder, file)
                shutil.copy2(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")


def main(src_folder, dest_folder):
    copy_jpg_files(src_folder, dest_folder)
    print(f"All .jpg files from {src_folder} have been copied to {dest_folder}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy .jpg files from source folder to destination folder, including subfolders. Throws exception if duplicate filenames are found.")
    parser.add_argument('-s', '--src_folder', help="Source folder to search for .jpg files", default='/content/gdrive/MyDrive/ML/ClearScanner/datasets')
    parser.add_argument('-d', '--dest_folder', help="Destination folder to copy .jpg files to", default='./dataset/images')

    args = parser.parse_args()

    main(args.src_folder, args.dest_folder)
