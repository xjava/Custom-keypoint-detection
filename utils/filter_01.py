import csv
import os
import shutil
from coco_dataset_util import image_ids_exists, read_coco, remove_image_ids, write_coco, file_names_to_image_ids, image_ids_to_file_names


def move_files(source_dir, destination_dir, filenames_list):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in filenames_list:
        source_file = os.path.join(source_dir, filename)
        if os.path.isfile(source_file):
            shutil.move(source_file, destination_dir)
            print(f"Moved: {filename}")
        else:
            print(f"File not found: {filename}")


def read_csv_to_list(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            for item in row:
                data.append(item)
        return data

coco_data1 = read_coco(
    "/Users/nikornlansa/Downloads/CordDetection.json")
coco_data2 = read_coco(
    "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/CordDetection.json")

ids1 = {image['id'] for image in coco_data1['images']}
ids2 = {image['id'] for image in coco_data2['images']}

set1 = set(ids1)
set2 = set(ids2)

diff = set1 - set2

l = [2, 17, 19, 27, 29, 31,33, 38,39, 40,46,50, 58, 66, 76, 79,80,84, 93, 99,101,112, 130, 133, 135,136,137, 147,149, 152,155, 158, 159, 186, 195, 206,
                                     221,174,175,184,189,190,191,198,210,255,250,288,290,295,296,
                                     233, 249, 272, 278, 281, 282, 329, 393, 429, 545, 605, 643, 661, 668, 699, 716,
                                     720, 721, 734, 737, 741, 749, 988,303,326,328,332,334,347,
                                     990, 993, 995, 1030, 1051, 1072, 1078, 1098, 1222, 1248, 1252,735,740,741,84,141,144,231,240,258,259,260,265,269,288,577]
print(f"{sorted(diff)}")


