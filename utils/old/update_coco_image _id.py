import json
import argparse
from coco_dataset_util import read_coco, write_coco

def modify_coco_ids(coco_file, output_file, start_image_id=1, start_annotation_id=1):
    # Load the COCO JSON file
    coco_data = read_coco(coco_file)

    # Create a mapping from old image_id to new image_id
    image_id_mapping = {image['id']: start_image_id + i for i, image in enumerate(coco_data['images'])}

    # Update image_id in images section
    for i, image in enumerate(coco_data['images']):
        image['id'] = start_image_id + i

    # Update image_id and category_id in annotations section
    for i, annotation in enumerate(coco_data['annotations']):
        # Map the old image_id to the new image_id
        annotation['image_id'] = image_id_mapping[annotation['image_id']]
        # Assign new category_id (assuming you want sequential category IDs)
        annotation['id'] = start_annotation_id + i

    # Save the modified COCO JSON file
    write_coco(output_file, coco_data)

def main():
    parser = argparse.ArgumentParser(description="Modify image_id and category_id in COCO JSON file")
    parser.add_argument("--coco_file", default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/MendeleyDetection.json',
                        help="Path to the input COCO JSON file (default: 'input_coco.json')")
    parser.add_argument("--output_file", default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/datasets/MendeleyDetection_update.json',
                        help="Path to the output COCO JSON file (default: 'output_coco.json')")
    parser.add_argument("--start_image_id", type=int, default=50001, help="Starting image_id number")
    parser.add_argument("--start_annotation_id", type=int, default=50001, help="Starting annotation_id number")

    args = parser.parse_args()

    modify_coco_ids(args.coco_file, args.output_file, args.start_image_id, args.start_annotation_id)

if __name__ == "__main__":
    main()
