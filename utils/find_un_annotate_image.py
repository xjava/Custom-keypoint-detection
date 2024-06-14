
import csv
import os
import shutil
from common_dataset_util import read_json
def copy_files(source_dir, destination_dir, filenames_list):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in filenames_list:
        source_file = os.path.join(source_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_dir)
            print(f"Copy: {filename}")
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

def get_image_files(coco_json_path):
    coco_data = read_json(coco_json_path)
    images_filenames = []
    for image in coco_data['images']:
        images_filenames.append(image['file_name'])
    return images_filenames

# Define the arrays
# array1 = [3,5,6,47,106,118,131,133,137,141,145,158,159,195,245,256,273,282,291,411,419,424,426,482,541,548,553,637,706,732,777,780,781,800,834,894,1035,1036,1066,1102]
# array2 = [2,3,5,6,26,44,47,89,101,118,131,133,141,145,163,195, 198,216,318,327,411,426,482,551,596,834,637,690,691,777, 779,800,858,927,1035,1066,11160,1168,1189,1191]
# array3 = [2,3,6,15,42,47,48,81,84,92,101,106,118,127,131,133,141,148,198,241,249,256,300,311,315,318,321,327,390,399,411,447,479,482,492,496,523,527,574,600,605,610,618,637,640,672,688,690,619,716,732,834,777,779,780,781,800,905,926,927,990,1001,1035,1066,1090,1264]

array1 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/un_annotate/DocumentDetection.txt")
array2 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/un_annotate/BoardDetection.txt")
array3 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/un_annotate/CardDetection.txt")
array4 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/un_annotate/DocumentDetection2.txt")


array1_annotate = get_image_files("/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/DocumentDetection.json")
array2_annotate = get_image_files("/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/BoardDetection.json")
array3_annotate = get_image_files("/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/CardDetection.json")
array4_annotate = get_image_files("/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/DocumentDetection2.json")


# array1 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/v3/tom.txt")
# array2 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/v3/dom.txt")
# array3 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/v3/pui.txt")


# array1 = [1837,1954,1955]
# array2 = [1708,2267,2273,2279,2284,2286,1814,2303,2304,1838,1847,1861,1867,2310,2312,1906,2315,1928,2009,2327,2329,2331,2387]
# array3 = [1731,1792,1837,1902,1907,1861,1879,1883,2024,2031,2032,2147,2148,2159,1954,2282,2494]

#remove empty string
array1 = filter(None, array1)
array2 = filter(None, array2)
array3 = filter(None, array3)
array4 = filter(None, array4)
# Convert arrays to sets
set1 = set(array1)
set2 = set(array2)
set3 = set(array3)
set4 = set(array4)

set1_annotate = set(array1_annotate)
set2_annotate = set(array2_annotate)
set3_annotate = set(array3_annotate)
set4_annotate = set(array4_annotate)


dif_1 = set1 - set1_annotate;
dif_2 = set2 - set2_annotate;
dif_3 = set3 - set3_annotate;
dif_4 = set4 - set4_annotate;

output_dir = '/Users/nikornlansa/Downloads/un_annotate'
copy_files('/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/DocumentDetection', output_dir, dif_1)
copy_files('/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/BoardDetection', output_dir, dif_2)
copy_files('/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/CardDetection', output_dir, dif_3)
copy_files('/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/DocumentDetection2', output_dir, dif_4)

