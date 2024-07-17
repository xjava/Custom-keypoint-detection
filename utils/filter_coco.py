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

    #เก่า remove_from_DocumentDetection = [2, 17, 19, 27, 29, 31, 33, 38, 39, 40, 46, 50, 58, 66, 76, 79, 80, 84, 93, 99, 101, 112, 130, 133, 135, 136, 137, 141, 144, 147, 149, 152, 155, 158, 159, 174, 175, 184, 186, 189, 190, 191, 195, 198, 206, 210, 221, 231, 233, 240, 249, 250, 255, 258, 259, 260, 265, 269, 272, 278, 281, 282, 288, 290, 295, 296, 303, 326, 328, 329, 332, 334, 347, 393, 429, 545, 577, 605, 643, 661, 668, 699, 716, 720, 721, 734, 735, 737, 740, 741, 749, 988, 990, 993, 995, 1030, 1051, 1072, 1078, 1098, 1222, 1248, 1252]

    remove_from_DocumentDetection = [84,141,144,231,240,258,259,260,265,269,577, 573,608,932,933,983,1015,1097, 1189,1098,1205] #มีห่วง
                                    #[1019, 1191]
    remove_from_DocumentDetection2 = [1708, 1740, 1745, 1764, 1809, 1810, 1889, 1890, 1938, 1962, 2015, 2016, 2283,
                                      2301, 2309, 2316, 2318, 2319, 2327, 2329, 2367]#done
    remove_from_CardDetection = [1538, 1575, 1686]#done
    remove_from_BoardDetection = [1287, 1310, 1320, 1358, 1406, 1411, 1413, 1430, 1447]

    remove_from_BookDetection = []

    remove_fromCordDetection = [2640, 2649, 2653, 2659, 2663, 2668, 2670, 2671, 2676, 2683, 2710, 2717, 2723, 2728, 2744, 2746, 2747, 2760, 2762,
     2763, 2765, 2768, 2775, 2791, 2792, 2796, 2800, 2804, 2814, 2817, 2819, 2845, 2854, 2855, 2861, 2865, 2867, 2868,
     2883, 2898, 2899, 2909, 2919, 2920, 2941, 2943, 2947, 2956, 2963, 2969, 2970, 2977, 2989, 2994, 3016, 3023, 3042,
     3055, 3062, 3081, 3084, 3120, 3122, 3138, 3142, 3146, 3147, 3150, 3151, 3156, 3162, 3218, 3219, 3235, 3256, 3263,
     3279, 3306, 3309, 3326, 3330, 3332, 3333, 3336, 3340, 3349, 3358, 3400, 3408, 3410, 3413, 3419, 3426, 3429, 3435,
     3451, 3456, 3465, 3471, 3474, 3487, 3491, 3495, 3497, 3499, 3514, 3516, 3524, 3525, 3547, 3549, 3562, 3571, 3578,
     3585, 3591, 3625]

    remove_from_MendeleyDetection = [50089, 50123, 50137, 50150, 50182, 50210, 50215, 50226, 50240, 50395, 50408, 50420,
                                     50459, 50498, 50516, 50566, 50590, 50627, 50684]
    remove_from_DocumentDetection3 = [3690, 3695, 3718, 3720, 3724, 3730, 3731, 3741, 3736, 3754, 3767,
                                      3780, 3783, 3791, 3798, 3815, 3818, 3819, 3829, 3842, 3847, 3855, 3867,
                                      3868, 3870,
                                      3871, 3872, 3873, 3879, 3880, 3887, 3888, 3889, 3901, 3905, 3908, 3909, 3911, 3914,
                                      3920, 3927, 3926, 3930, 3932, 3933, 3945, 3947, 3948, 3967, 3971, 3974, 3980,
                                      3983,
                                      4001, 4002, 4007, 4026, 4032, 4044, 4069, 4073, 4082, 4084, 4087, 4088, 4089,
                                      4090, 4096, 4097, 4113, 4126, 4127, 4151, 4153, 4159, 4243, 4261, 4276, 4278,
                                      4281, 4283, 4294, 4318,
                                      4341, 4377, 4379, 4380, 4381, 4393, 4475, 4492, 4495, 4516, 4522, 4524, 4526, 4571,
                                      4649, 4702, 4709, 4711, 4712, 4730, 4731, 4744, 4746, 4747, 4759, 4762, 4770,
                                      4780, 4781, 4785, 4787,
                                      4788, 4792, 4795, 4801, 4802, 4803, 4805, 4807, 4811, 4812, 4813, 4814, 4816,
                                      4819, 4828, 4813, 4832, 4833, 4834, 4835, 4836, 4842, 4844, 4846, 4847, 4848,
                                      4850, 4876, 4878, 4887,
                                      4892, 4894, 4898, 4908, 4909, 4924, 4931, 4940, 4941, 4943, 4944, 4949, 4950,
                                      4952, 4962, 4971, 4976,
                                      4986, 4987, 4988, 4989, 4992, 4994, 4995, 4999, 5001, 5002, 5011, 5012, 5015,
                                      5013, 5016, 5017, 5018, 5019, 5020, 5022, 5024, 5025, 5026, 5041,
                                      5088, 5010, 5112, 5116, 5117, 5120, 5122, 5147, 5162, 5163, 5165, 5167, 5169,
                                      5170, 5183, 5194, 5196, 5201, 5202, 5204, 5260, 5225, 5234, 5239,
                                      5244, 5248, 5253, 5312, 5327, 5328, 5351, 5357]

    # ex IMG_01_01-02865.JPG, IMG_01_01-02866.JPG IMG_01_01-02868.JPG IMG_01_01-03025.JPG IMG_01_01-03032.JPG IMG_01_01-03254.JPG
    # ภาพไม่มีคุณภาพ 4986,4987,4988,4989,4992,4994,4995,4999,5001,5002,5011,5012,5015,5013,5016,5017,5018,5019,5020,5022,5024,5025,5026,5041

    # **** จริงๆต้องเอา 133 ออก ไม่ใช่ 113
    # filter_out_image_ids(
    #     "/Users/nikornlansa/Downloads/DocumentDetection-80.json",
    #     remove_from_DocumentDetection)
    # filter_out_image_ids(
    #     "/Users/nikornlansa/Downloads/BoardDetection-77.json",
    #     remove_from_BoardDetection)
    # filter_out_image_ids(
    #     "/Users/nikornlansa/Downloads/CardDetection-81.json",
    #     remove_from_CardDetection)
    # filter_out_image_ids(
    #     "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/BookDetection.json",
    #     remove_from_BookDetection)
    filter_out_image_ids(
        "/Users/nikornlansa/Downloads/DocumentDetection2-84.json",
        remove_from_DocumentDetection2)
    # filter_out_image_ids(
    #     "/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/MendeleyDetection.json",
    #     remove_from_MendeleyDetection)
    #filter_out_image_ids("/Users/nikornlansa/Downloads/DocumentDetection3-82.json", remove_from_DocumentDetection3)
    #filter_out_image_ids("/Users/nikornlansa/Downloads/CordDetection.json", remove_fromCordDetection)

