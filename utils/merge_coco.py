import json
import copy
import argparse
import os
import shutil
from coco_dataset_util import read_coco, write_coco
import random
def merge_coco_files(file_paths, output_file, category_mapping, shuffle, merge_image):
    category_id_mapping = parse_category_mapping(category_mapping)
    global out_filename
    if merge_image:
        parent = os.path.dirname(output_file)
        out_filename, ext = os.path.splitext(os.path.basename(output_file))
        images_folder = os.path.join(parent, out_filename)
        if os.path.exists(images_folder):
            shutil.rmtree(images_folder)
        os.makedirs(images_folder)

    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_ids = set()
    annotation_ids = set()

    new_category_id_mapping = {}

    seen_files = set()
    for file_path in file_paths:
        coco_data = read_coco(file_path)
        ds_filename, ds_ext = os.path.splitext(os.path.basename(file_path))
        # Merge categories and create a mapping from old category IDs to new ones
        for category in coco_data['categories']:
            old_category_id = category['id']
            if category_id_mapping and old_category_id in category_id_mapping:
                new_category_id = category_id_mapping[old_category_id]
            else:
                if old_category_id not in new_category_id_mapping:
                    new_category_id = len(new_category_id_mapping) + 1
                    new_category_id_mapping[old_category_id] = new_category_id
                else:
                    new_category_id = new_category_id_mapping[old_category_id]

            if not any(cat['id'] == new_category_id for cat in merged_data['categories']):
                new_category = copy.deepcopy(category)
                new_category['id'] = new_category_id
                merged_data['categories'].append(new_category)

        # Merge images and annotations
        for image in coco_data['images']:
            if image['id'] in image_ids:
                raise ValueError(f"Duplicate image id {image['id']} found in file {file_path}")
            image_ids.add(image['id'])
            new_image = copy.deepcopy(image)
            merged_data['images'].append(new_image)
            # nikorn
            new_image['annotated'] = True
            new_image['num_annotations'] = 1
            new_image['category_ids'] = [1]

            # nikorn
            if merge_image:
                file_name = new_image['file_name']
                if file_name in seen_files:
                    raise Exception(f"Duplicate file found: {file_name}")
                seen_files.add(file_name)
                src_path = os.path.join(os.path.dirname(file_path), ds_filename, file_name)
                dest_path = os.path.join(os.path.dirname(output_file), out_filename, file_name)
                shutil.copy2(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")

        for annotation in coco_data['annotations']:
            if annotation['id'] in annotation_ids:
                raise ValueError(f"Duplicate annotation id {annotation['id']} found in file {file_path}")
            annotation_ids.add(annotation['id'])
            new_annotation = copy.deepcopy(annotation)
            new_annotation['category_id'] = category_id_mapping.get(annotation['category_id'],
                                                                    annotation['category_id'])
            merged_data['annotations'].append(new_annotation)

    if shuffle:
        # Shuffle images and annotations
        random.shuffle(merged_data['images'])
        random.shuffle(merged_data['annotations'])

    write_coco(output_file,merged_data)
    print(f'Merged {len(file_paths)} files into {output_file}')


def parse_category_mapping(mapping_str):
    if mapping_str is None:
        return {}
    mapping = {}
    pairs = mapping_str.split(',')
    for pair in pairs:
        old_id, new_id = map(int, pair.split(':'))
        mapping[old_id] = new_id
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple COCO JSON files into one with optional category ID remapping.")
    parser.add_argument('-i', '--input', nargs='+', help="List of input COCO JSON files", required=False,
                        default=['/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/DocumentDetection.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/DocumentDetection2.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/BoardDetection.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/CardDetection.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/BookDetection.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/MendeleyDetection.json',
                                 '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/CordDetection.json'])
    # parser.add_argument('-o', '--output', help="Output file name", required=False, default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/all.json')
    parser.add_argument('-o', '--output', help="Output file name", required=False,
                        default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/images.json')
    parser.add_argument('--merge_image', type=bool, default=True)
    parser.add_argument('-m', '--mapping', help="Category ID mapping in the form 'old_id1:new_id1,old_id2:new_id2,...'",
                        required=False, default="2:1,3:1,7:1")

    args = parser.parse_args()

    merge_coco_files(args.input, args.output, args.mapping, False, args.merge_image)

# python merge_coco_(change_id).py -i coco1.json coco2.json coco3.json -o merged_coco.json -m "2:1,3:1,7:1"
