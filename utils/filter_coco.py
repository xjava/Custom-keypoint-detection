import os
from coco_dataset_util import read_coco, remove_image_ids, write_coco, image_ids_exists


def filter_out_image_ids(coco_path, image_ids):
    parent = os.path.dirname(coco_path)
    filename, ext = os.path.splitext(os.path.basename(coco_path))
    filtered_path = os.path.join(parent, f"{filename}_filtered.json")

    coco = read_coco(coco_path)

    found_image_ids, missing_image_ids = image_ids_exists(coco, image_ids)
    if len(missing_image_ids) > 0:
        raise Exception(f"unkone image ids found: {missing_image_ids}")
    coco = remove_image_ids(coco, image_ids)
    write_coco(filtered_path, coco)


if __name__ == "__main__":

    158 605 661 668 721 737 741 393
    remove_from_DocumentDetection = [17, 27, 29, 31, 50, 58, 66, 76, 79, 93, 112, 130, 113, 136, 186, 195, 206, 221,
                                     233, 249, 272, 278, 281, 282, 329, 429, 545, 643, 699, 716, 720, 734, 749, 988,
                                     990, 993, 995, 1030, 1051, 1072, 1078, 1098, 1222, 1248, 1252]
    remove_from_BoardDetection = [1287, 1310, 1320, 1358, 1406, 1411, 1413, 1430, 1447]
    remove_from_CardDetection = [1524, 1533, 1538, 1553, 1591, 1685, 1686, 1689]
    remove_from_BookDetection = []
    remove_from_DocumentDetection2 = [1708, 1740, 1745, 2279, 1764, 2283, 1809, 1810, 1824, 2301, 2306, 2309, 1890,
                                      1909, 2316, 1938, 1962, 2038, 2053, 2054, 2367, 2381, 2390, 2404, 2408, 2318,
                                      2319, 1824]

    #**** จริงๆต้องเอา 133 ออก ไม่ใช่ 113
    filter_out_image_ids(
        "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/DocumentDetection.json",
        remove_from_DocumentDetection)
    filter_out_image_ids(
        "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/BoardDetection.json",
        remove_from_BoardDetection)
    filter_out_image_ids(
        "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/CardDetection.json",
        remove_from_CardDetection)
    filter_out_image_ids(
        "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/BookDetection.json",
        remove_from_BookDetection)
    filter_out_image_ids(
        "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/DocumentDetection2.json",
        remove_from_DocumentDetection2)
