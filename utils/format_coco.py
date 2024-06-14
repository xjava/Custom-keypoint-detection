from common_dataset_util import list_files
from coco_dataset_util import  read_coco, write_coco
import os
coco_dir = '/Users/nikornlansa/Workspace/ClearScanner/documentcornerlocalization/dataset'
coco_files = list_files(coco_dir, '.json')
for coco_file in coco_files:
    coco_path = os.path.join(coco_dir, coco_file)
    coco_data = read_coco(coco_path)
    write_coco(coco_path, coco_data)
