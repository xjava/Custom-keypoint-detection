import os
import shutil
from search_duplicate_image import search_duplicate_image

def move_files(filepaths, destination_dir, ):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filepath in filepaths:
        if os.path.isfile(filepath):
            shutil.move(filepath, destination_dir)
            print(f"Moved: {filepath}")
        else:
            print(f"File not found: {filepath}")


if __name__ == '__main__':
    image_dirs = ['/Users/nikornlansa/Workspace/ML/ClearScanner/Images-resize/Document/01']
    coco_paths = ['/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/DocumentDetection.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/BoardDetection.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/CardDetection.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/DocumentDetection2.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/BookDetection.json'
                  ]
    dup_path = '/Users/nikornlansa/Workspace/ML/ClearScanner/Images-resize-duplicated2'
    to_remove_file_paths = search_duplicate_image(image_dirs, coco_paths)
    move_files(to_remove_file_paths, dup_path)
