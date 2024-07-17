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



array1 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/doc1/tom_filter.txt")
array2 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/doc1/dom_filter.txt")
#array3 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/doc1/pui_filter.txt")
array3 = []

# remove empty string
array1 = filter(None, array1)
array2 = filter(None, array2)
array3 = filter(None, array3)


coco_data = read_coco(
    "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/DocumentDetection.json")


array1 = file_names_to_image_ids(coco_data, array1)
array2 = file_names_to_image_ids(coco_data, array2)
array3 = [2, 17, 19, 27, 29, 31,33, 38,39, 40,46,50, 58, 66, 76, 79,80,84, 93, 99,101,112, 130, 133, 135,136,137, 147,149, 152,155, 158, 159, 186, 195, 206,
                                     221,174,175,184,189,190,191,198,210,255,250,288,290,295,296,
                                     233, 249, 272, 278, 281, 282, 329, 393, 429, 545, 605, 643, 661, 668, 699, 716,
                                     720, 721, 734, 737, 741, 749, 988,303,326,328,332,334,347,
                                     990, 993, 995, 1030, 1051, 1072, 1078, 1098, 1222, 1248, 1252]



# Convert arrays to sets
set1 = set(array1)
set2 = set(array2)
set3 = set(array3)
# set4 = set(array4)
set_all = set1 | set2 | set3  # | set4

# 1. Find numbers repeated in every array
common_all = set1 & set2 & set3

# 2. Find numbers that are duplicates in only 2 arrays and are not in common_all
common_two = ((set1 & set2) | (set2 & set3) | (set3 & set1)) - common_all

# 3. Find unique numbers and in which array they are
unique_array1 = set1 - (set2 | set3)
unique_array2 = set2 - (set1 | set3)
unique_array3 = set3 - (set1 | set2)

# Convert the results back to lists (if needed)
common_all_list = sorted(list(common_all))
common_two_list = sorted(list(common_two))
unique_array1_list = sorted(list(unique_array1))
unique_array2_list = sorted(list(unique_array2))
unique_array3_list = sorted(list(unique_array3))

# Print the results
print(f"ทั้งหมด: {len(set_all)} ภาพ")
print("ทั้งหมด", sorted(set_all))
# print(f"ต้อม ทั้งหมด: {len(set1)} ภาพ")
# print(f"ดม ทั้งหมด: {len(set2)} ภาพ")
# print(f"ปุย ทั้งหมด: {len(set3)} ภาพ")
# print(f"เหมือนกัน 3 คน :{len(common_all_list)} ภาพ")
# print(f"ที่เห็นเหมือนกัน 2 คน :{len(common_two_list)} ภาพ")
# print(f"ต้อมคนเดียว :{len(unique_array1_list)} ภาพ")
# print(f"ดมคนเดียว :{len(unique_array2_list)} ภาพ")
# print(f"ปุยคนเดียว :{len(unique_array3_list)} ภาพ")
#
# print("เหมือนกัน 3 คน: ", common_all_list)
# print("ที่เห็นเหมือนกัน 2 คน:", common_two_list)
# print("ต้อมคนเดียว:", unique_array1_list)
# print("ดมคนเดียว:", unique_array2_list)
# print("ปุยคนเดียว:", unique_array3_list)

# coco_data = read_coco('/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/CordDetection.json')

# found_image_ids, missing_image_ids = image_ids_exists(coco_data, set_all)
#
# filtered_coco_data = remove_image_ids(coco_data, found_image_ids)

# write_coco('/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/CordDetection_filter.json', filtered_coco_data)

# source_directory = '/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/coco/v3/doc_v3_all'
# destination_directory = '/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/coco/v3/doc_v3_all_selected_2'
# move_files(source_directory, destination_directory, common_two_list)
