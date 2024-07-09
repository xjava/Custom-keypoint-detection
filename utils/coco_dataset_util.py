import os
import json
import numpy as np
import shutil
from common_dataset_util import read_json, write_json


def read_coco(coco_path):
    print(f"Reading coco from path: {coco_path}")
    coco_data = read_json(coco_path)
    print(
        f"Total images:{len(coco_data['images'])} annotations: {len(coco_data['annotations'])} categories: {len(coco_data['categories'])}")
    return coco_data


def write_coco(coco_path, coco_data):
    print(f"Writing coco to path: {coco_path}")
    write_json(coco_path, coco_data)
    print(
        f"Total images:{len(coco_data['images'])} annotations: {len(coco_data['annotations'])} categories: {len(coco_data['categories'])}")


def image_ids_exists(coco_data, image_ids_to_check):
    # Extract image_ids from the 'images' section
    existing_image_ids = {image['id'] for image in coco_data['images']}  # Using a set for fast lookup

    # Check if the given image_ids are in the existing_image_ids
    found_image_ids = [image_id for image_id in image_ids_to_check if image_id in existing_image_ids]
    missing_image_ids = [image_id for image_id in image_ids_to_check if image_id not in existing_image_ids]

    # Print the results
    print("Found image IDs:")
    print(found_image_ids)
    print("\nMissing image IDs:")
    print(missing_image_ids)
    return found_image_ids, missing_image_ids


def image_names_exists(coco_data, file_names_to_check):
    # Extract file_names from the 'images' section
    existing_file_names = {image['file_name'] for image in coco_data['images']}  # Using a set for fast lookup

    # Check if the given file_names are in the existing_file_names
    found_file_names = [file_name for file_name in file_names_to_check if file_name in existing_file_names]
    missing_file_names = [file_name for file_name in file_names_to_check if file_name not in existing_file_names]

    # Print the results
    print("Found file names:")
    print(found_file_names)
    print("\nMissing file names:")
    print(missing_file_names)
    return found_file_names, missing_file_names


def remove_image_ids(coco_data, image_ids_to_remove):
    # Filter the 'images' section to remove specified image_ids
    filtered_images = [image for image in coco_data['images'] if image['id'] not in image_ids_to_remove]

    # Filter the 'annotations' section to remove annotations with specified image_ids
    filtered_annotations = [annotation for annotation in coco_data['annotations'] if
                            annotation['image_id'] not in image_ids_to_remove]

    # Optionally, filter other sections like 'categories', if needed
    # In most cases, 'categories' might remain the same, but it depends on the specific use case
    filtered_categories = coco_data['categories']

    # Construct the filtered COCO JSON data
    filtered_coco_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories,
        # Copy other sections if they exist
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }

    return filtered_coco_data


def keep_only_image_ids(coco_data, image_ids_to_keep):
    # Filter the 'images' section
    filtered_images = [image for image in coco_data['images'] if image['id'] in image_ids_to_keep]

    # Filter the 'annotations' section
    filtered_annotations = [annotation for annotation in coco_data['annotations'] if
                            annotation['image_id'] in image_ids_to_keep]

    # Optionally, filter other sections like 'categories', if needed
    # In most cases, 'categories' might remain the same, but it depends on the specific use case
    filtered_categories = coco_data['categories']

    # Construct the filtered COCO JSON data
    filtered_coco_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories,
        # Copy other sections if they exist
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }

    return filtered_coco_data


def file_names_to_image_ids(coco_data, file_names):
    # Create a mapping from file_name to id
    file_name_to_id = {image['file_name']: image['id'] for image in coco_data['images']}

    # Convert the list of file_names to list of ids
    image_ids = [file_name_to_id[file_name] for file_name in file_names if file_name in file_name_to_id]

    return image_ids


def image_ids_to_file_names(coco_data, image_ids):
    # Create a mapping from id to file_name
    id_to_file_name = {image['id']: image['file_name'] for image in coco_data['images']}

    # Convert the list of ids to list of file_names
    file_names = [id_to_file_name[image_id] for image_id in image_ids if image_id in id_to_file_name]

    return file_names
