import difPy
import os
from common_dataset_util import sort_path_by_filename
from common_dataset_util import read_json


def get_image_id_by_file_name(coco_datas, file_name):
    # Search for the image with the given file name
    for coco_data in coco_datas:
        for image in coco_data['images']:
            if image['file_name'] == file_name:
                return image['id']
    # If the file name is not found, return None
    return None


def search_duplicate_image(image_dirs, search_coco_paths):
    dif = difPy.build(image_dirs)
    se = difPy.search(dif)
    result_keys = se.result.keys()
    dup_paths_list = []

    for result_key in result_keys:
        dup = [result_key]
        for s in se.result[result_key]:
            dup.append(s[0])
        sort_path_by_filename(dup)
        #print(f'found duplicated: {dup}')
        dup_paths_list.append(dup)

    coco_datas = []
    for path in search_coco_paths:
        coco_datas.append(read_json(path))

    dup_files_list = []
    dup_annotate_files_list = []
    to_remove = []
    to_delete_from_dataset = []
    for dup_paths in dup_paths_list:
        to_remove.extend(dup_paths[1:])
        image_ids = []
        annotate_dub = True
        for dup_path in dup_paths:
            file_name = os.path.basename(dup_path)
            image_id = get_image_id_by_file_name(coco_datas, file_name)
            if image_id is None:
                annotate_dub = False
            image_ids.append([file_name, image_id])
        dup_files_list.append(image_ids)
        image_ids = []
        if annotate_dub:
            for dup_path in dup_paths:
                file_name = os.path.basename(dup_path)
                image_id = get_image_id_by_file_name(coco_datas, file_name)
                image_ids.append(image_id)
            dup_annotate_files_list.append(image_ids)
            to_delete_from_dataset.extend(image_ids[1:])

    print(f'\n dup paths {len(dup_paths_list)}: {dup_paths_list}')
    print(f'\n dup files {len(dup_files_list)}: {dup_files_list}')
    print(f'\n image paths to remove {len(to_remove)}: {to_remove}')
    print(f'\n dup image_ids {len(dup_annotate_files_list)}: {dup_annotate_files_list}')
    print(f'\n image ids to delete from data set {len(to_delete_from_dataset)}: {to_delete_from_dataset}')
    return to_remove

if __name__ == '__main__':
    image_dirs = ['/Users/nikornlansa/Workspace/ML/ClearScanner/Images-resize']
    coco_paths = ['/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/DocumentDetection.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/DocumentDetection2.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/BoardDetection.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/CardDetection.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/BookDetection.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/MendeleyDetection.json',
                  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/CordDetection.json',
                  ]

    search_duplicate_image(image_dirs, coco_paths)
