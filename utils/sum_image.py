
import csv
import os
import shutil
from coco_dataset_util import image_ids_exists, read_coco, remove_image_ids, write_coco
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



# Define the arrays
# array1 = [3,5,6,47,106,118,131,133,137,141,145,158,159,195,245,256,273,282,291,411,419,424,426,482,541,548,553,637,706,732,777,780,781,800,834,894,1035,1036,1066,1102]
# array2 = [2,3,5,6,26,44,47,89,101,118,131,133,141,145,163,195, 198,216,318,327,411,426,482,551,596,834,637,690,691,777, 779,800,858,927,1035,1066,11160,1168,1189,1191]
# array3 = [2,3,6,15,42,47,48,81,84,92,101,106,118,127,131,133,141,148,198,241,249,256,300,311,315,318,321,327,390,399,411,447,479,482,492,496,523,527,574,600,605,610,618,637,640,672,688,690,619,716,732,834,777,779,780,781,800,905,926,927,990,1001,1035,1066,1090,1264]

# array1 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/mendeley/tom.txt")
# array2 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/mendeley/dom.txt")
# array3 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/mendeley/pui.txt")
# array4 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/mendeley/tu.txt")

# array1 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/v3/tom.txt")
# array2 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/v3/dom.txt")
# array3 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/v3/pui.txt")

# array1 = [3333,3263,3330,3016,2899,3142,3309,3426,2817,2747,3571,2796,3549,3147,3349,2909,3150,2943,3219,3514,2649,3491,2800,3516,3419,3218,2970,3156,3055,3023,3279,3062,3547,3413,2854,2746,3429,3578,2760,3495,2919,3235,3042,2898,3471,3465,2659,3474,3435,3340,3326,2792]
# array2 = [2668,2717,3330,3426,3571,2768,2710,3150,2683,3516,2744,3547,2653,3525,2855,3562,2861,3471,2663,3474,2775,2791,3435,2868,3340,2792,3410,3400]
# array3 = [2723,2668,2747,2762, 2683,2765,3263,3104,3426,2747,2768,2710, 3156,2947,2653,2855,2819,2861,2920,2989,2956,2663,2868,2640,3340,2792,2670,2671,2728,2814,3571,3349,2969,3219,2649,2763]
# array4 = [3625, 3591, 3138, 3120, 3306, 3162, 2865, 3499, 3456, 3524, 3146, 3256, 3084, 3151, 3585,2804,3487,2883,2941,3358,2994,3081,2956,3122,2676,3451,2977,3332,3336, 2845, 3408, 2867,2963
# ,3497]

array1 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/doc1/tom_filter.txt")
array2 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/doc1/dom_filter.txt")
array3 = read_csv_to_list("/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/clean/doc1/pui_filter.txt")



#remove empty string
array1 = filter(None, array1)
array2 = filter(None, array2)
array3 = filter(None, array3)



# Convert arrays to sets
set1 = set(array1)
set2 = set(array2)
set3 = set(array3)
#set4 = set(array4)
set_all = set1 | set2 | set3 #| set4

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
print(f"ต้อม ทั้งหมด: {len(set1)} ภาพ")
print(f"ดม ทั้งหมด: {len(set2)} ภาพ")
print(f"ปุย ทั้งหมด: {len(set3)} ภาพ")
print(f"เหมือนกัน 3 คน :{len(common_all_list)} ภาพ")
print(f"ที่เห็นเหมือนกัน 2 คน :{len(common_two_list)} ภาพ")
print(f"ต้อมคนเดียว :{len(unique_array1_list)} ภาพ")
print(f"ดมคนเดียว :{len(unique_array2_list)} ภาพ")
print(f"ปุยคนเดียว :{len(unique_array3_list)} ภาพ")


print("เหมือนกัน 3 คน: ", common_all_list)
print("ที่เห็นเหมือนกัน 2 คน:", common_two_list)
print("ต้อมคนเดียว:", unique_array1_list)
print("ดมคนเดียว:", unique_array2_list)
print("ปุยคนเดียว:", unique_array3_list)

#coco_data = read_coco('/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/CordDetection.json')

# found_image_ids, missing_image_ids = image_ids_exists(coco_data, set_all)
#
# filtered_coco_data = remove_image_ids(coco_data, found_image_ids)

#write_coco('/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/CordDetection_filter.json', filtered_coco_data)

# source_directory = '/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/coco/v3/doc_v3_all'
# destination_directory = '/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/dataset/coco/v3/doc_v3_all_selected_2'
# move_files(source_directory, destination_directory, common_two_list)