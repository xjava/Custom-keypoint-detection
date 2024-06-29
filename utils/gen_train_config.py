import os.path

from merge_coco import merge_coco_files
from common_dataset_util import write_json, read_json


def create_dataset_split_info(coco_path, train_path, val_path, test_path):
    return {
        "all": len(read_json(coco_path)['images']),
        "train": len(read_json(train_path)['images']),
        "val": len(read_json(val_path)['images']),
        "test": len(read_json(test_path)['images'])
    }


# config
info = {'version': 2,
          # 'datasets': ["DocumentDetection", "DocumentDetection2", "BoardDetection", "CardDetection", "BookDetection",
          #              "MendeleyDetection", "CordDetection"],
        'datasets': ["DocumentDetection", "DocumentDetection2", "BoardDetection", "CardDetection", "BookDetection"],
          'category_mapping': "2:1,3:1,7:1"
          }

if __name__ == '__main__':
    dataset_dir = '/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/dataset'
    config_dir = os.path.join('/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/train_config',
                              f'version_{info["version"]}')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    datasets = info["datasets"]
    category_mapping = info["category_mapping"]

    coco_paths = []
    train_paths = []
    val_paths = []
    test_paths = []
    dataset_split_infos = []
    coco_count = 0
    train_count = 0
    val_count = 0
    test_count = 0
    for dataset in datasets:
        coco_path = os.path.join(dataset_dir, f'{dataset}.json')
        train_path = os.path.join(dataset_dir, f'{dataset}_train.json')
        val_path = os.path.join(dataset_dir, f'{dataset}_val.json')
        test_path = os.path.join(dataset_dir, f'{dataset}_test.json')
        train_paths.append(train_path)
        val_paths.append(val_path)
        test_paths.append(test_path)
        split_info = create_dataset_split_info(coco_path, train_path, val_path, test_path)
        coco_count += split_info["all"]
        train_count += split_info["train"]
        val_count += split_info["val"]
        test_count += split_info["test"]
        dataset_split_infos.append({dataset: split_info})

    info["datasets"] = dataset_split_infos
    info["total"] = {
        "all": coco_count,
        "train": train_count,
        "val": val_count,
        "test": test_count
    }
    merge_coco_files(train_paths, os.path.join(config_dir, 'train_data.json'), category_mapping, True, False)
    merge_coco_files(val_paths, os.path.join(config_dir, 'validation_data.json'), category_mapping, True, False)
    merge_coco_files(test_paths, os.path.join(config_dir, 'test_data.json'), category_mapping, True, False)

    write_json(os.path.join(config_dir, 'info.json'), info)


        # # all_paths = [os.path.join(dataset_dir, f'{file}.json') for file in datasets]
        # train_paths = [os.path.join(dataset_dir, f'{file}_train.json') for file in datasets]
        # val_paths = [os.path.join(dataset_dir, f'{file}_val.json') for file in datasets]
        # test_paths = [os.path.join(dataset_dir, f'{file}_test.json') for file in datasets]

        # merge_coco_files(all_paths, os.path.join(config_dir, 'all.json'), category_mapping, True, False)


