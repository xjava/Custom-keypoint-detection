import json
import copy
import argparse


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def merge_coco_files(file_paths, output_file, category_id_mapping=None):
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    next_image_id = 1
    next_annotation_id = 1
    new_category_id_mapping = {}

    for file_path in file_paths:
        coco_data = load_json(file_path)

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
            new_image = copy.deepcopy(image)
            old_image_id = image['id']
            new_image_id = next_image_id
            new_image['id'] = new_image_id
            next_image_id += 1
            merged_data['images'].append(new_image)

            # Update annotations with new image and category IDs
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == old_image_id:
                    new_annotation = copy.deepcopy(annotation)
                    new_annotation['id'] = next_annotation_id
                    new_annotation['image_id'] = new_image_id
                    new_annotation['category_id'] = category_id_mapping.get(annotation['category_id'],
                                                                            annotation['category_id'])
                    next_annotation_id += 1
                    merged_data['annotations'].append(new_annotation)

    save_json(merged_data, output_file)


def parse_category_mapping(mapping_str):
    if mapping_str is None:
        return {}
    mapping = {}
    pairs = mapping_str.split(',')
    for pair in pairs:
        old_id, new_id = map(int, pair.split(':'))
        mapping[old_id] = new_id
    return mapping


def main(coco_files=None, output_file=None, category_mapping=None):
    if coco_files is None:
        coco_files = ['../dataset/DocumentDetection.json', '../dataset/BoardDetection.json', '../dataset/CardDetection.json']
    if output_file is None:
        output_file = '../dataset/all.json'

    category_id_mapping = parse_category_mapping(category_mapping)
    merge_coco_files(coco_files, output_file, category_id_mapping)
    print(f'Merged {len(coco_files)} files into {output_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple COCO JSON files into one with optional category ID remapping.")
    parser.add_argument('-i', '--input', nargs='+', help="List of input COCO JSON files", required=False)
    parser.add_argument('-o', '--output', help="Output file name", required=False)
    parser.add_argument('-m', '--mapping', help="Category ID mapping in the form 'old_id1:new_id1,old_id2:new_id2,...'",
                        required=False)

    args = parser.parse_args()

    main(coco_files=args.input, output_file=args.output, category_mapping=args.mapping)

#python merge_coco.py -i coco1.json coco2.json coco3.json -o merged_coco.json -m "2:1,3:2"
