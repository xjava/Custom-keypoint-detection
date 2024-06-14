import os.path

from merge_coco import merge_coco_files
from common_dataset_util import write_json
#config
config = {'version': 1,
          'datasets': ["DocumentDetection", "DocumentDetection2", "BoardDetection", "CardDetection", "BookDetection", "MendeleyDetection", "CordDetection"],
          'category_mapping': "2:1,3:1,7:1"
          }


dataset_dir = '/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/dataset'
config_dir = os.path.join('/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/train_config', f'version_{config["version"]}')
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

datasets = config["datasets"]
category_mapping = config["category_mapping"]
train_paths = [os.path.join(dataset_dir, f'{file}_train.json') for file in datasets]
val_paths = [os.path.join(dataset_dir, f'{file}_val.json') for file in datasets]
test_paths = [os.path.join(dataset_dir, f'{file}_test.json') for file in datasets]

#โค้ดการ merge id จะวิ่ง 1 ใหม่
next_image_id = 1
next_annotation_id = 1
next_image_id, next_annotation_id = merge_coco_files(train_paths, os.path.join(config_dir, 'train.json'), category_mapping, next_image_id, next_annotation_id, False)
next_image_id, next_annotation_id = merge_coco_files(val_paths, os.path.join(config_dir, 'val.json'), category_mapping, next_image_id, next_annotation_id, False)
next_image_id, next_annotation_id = merge_coco_files(test_paths, os.path.join(config_dir, 'test.json'), category_mapping, next_image_id, next_annotation_id, False)
write_json( os.path.join(config_dir, 'config.json'), config)


