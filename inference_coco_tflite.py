#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 23:28:19 2022

@author: a975193
"""

import cv2
import time
import glob
import numpy as np
import tensorflow.compat.v1 as tf
import os
import os
import argparse
import cv2
from utils.common_dataset_util import read_json
from utils.geometry_util import calculate_polygon_iou, calculate_keypoints_distance_mse, calculate_distance_point
import csv


class InferenceResult:
    image_id = ""
    file_name = ""
    width = 0
    height = 0
    iou_percentage = 0
    distance_by_ref_mse = 0
    distance_by_ref_p1 = 0
    distance_by_ref_p2 = 0
    distance_by_ref_p3 = 0
    distance_by_ref_p4 = 0
    score_inference_p1 = 0
    score_inference_p2 = 0
    score_inference_p3 = 0
    score_inference_p4 = 0
    inference_p1x = 0
    inference_p1y = 0
    inference_p2x = 0
    inference_p2y = 0
    inference_p3x = 0
    inference_p3y = 0
    inference_p4x = 0
    inference_p4y = 0
    ground_truth_p1x = 0
    ground_truth_p1y = 0
    ground_truth_p2x = 0
    ground_truth_p2y = 0
    ground_truth_p3x = 0
    ground_truth_p3y = 0
    ground_truth_p4x = 0
    ground_truth_p4y = 0


def list_files(directory, extensions):
    # List all files in the given directory
    files = os.listdir(directory)
    # Filter out the files that end with .csv

    filtered_files = [file for file in files if file.endswith(extensions)]
    return filtered_files


CV_RED = (0, 0, 255)
CV_GREEN = (0, 255, 0)
CV_MAGENTA = (255, 0, 255)
CV_BLUE = (255, 0, 0)
colors = [CV_RED, CV_GREEN, CV_MAGENTA,
          CV_BLUE]
ground_truth_color = CV_GREEN
inference_color = CV_RED


# cv2.namedWindow("display", cv2.WINDOW_NORMAL)

def get_annotations_by_image_id(image_id, coco_json):
    annotations = []
    for annotation in coco_json['annotations']:
        if annotation['image_id'] == image_id:
            annotations.append(annotation)
    return annotations


def draw_ground_truth(image, w, h, image_json, coco_json):
    image_id = image_json['id']
    annotations = get_annotations_by_image_id(image_id, coco_json)

    fontScale = 2
    adjust = 60
    circle_size = 14
    border_thickness = 2
    if h < 512 and w < 512:
        fontScale = 1
        adjust = 30
        circle_size = 7
        border_thickness = 1

    for annotation in annotations:
        points = []
        if 'keypoints' in annotation:
            keypoints = annotation['keypoints']
            index = 0
            for i in range(0, len(keypoints), 3):
                kp_x = keypoints[i]
                kp_y = keypoints[i + 1]
                v = keypoints[i + 2]  # Visibility flag
                if v > 0:  # Draw keypoint if visible
                    cv2.circle(image, (kp_x, kp_y), circle_size, ground_truth_color, -1)
                    points.append((kp_x, kp_y))
                    index = index + 1
        # Draw the line
        # border_color = (0, 255, 0)
        cv2.line(image, points[0], points[1], ground_truth_color, border_thickness)
        cv2.line(image, points[1], points[2], ground_truth_color, border_thickness)
        cv2.line(image, points[2], points[3], ground_truth_color, border_thickness)
        cv2.line(image, points[3], points[0], ground_truth_color, border_thickness)
        cv2.putText(
            image,  # image on which to draw text
            f"ID : {image_id}",
            (int(w / 2) - 120, adjust),  # bottom left corner of text
            cv2.FONT_HERSHEY_SIMPLEX,  # font to use
            fontScale,  # font scale
            (0, 255, 0),  # color
            5,  # line thickness
        )
        return points
    return None


def draw_inference(img, kp, kp_s, h, w):
    fontScale = 2
    adjust = 60
    circle_size = 10
    border_thickness = 2
    if h < 512 and w < 512:
        fontScale = 1
        adjust = 30
        circle_size = 5
        border_thickness = 1

    points = []

    for index, kp_data in enumerate(kp):
        x = int(kp_data[1] * w)
        y = int(kp_data[0] * h)
        points.append((x, y))
        cv2.circle(img, (x, y), circle_size, inference_color, -1)
    cv2.line(img, points[0], points[1], inference_color, border_thickness)
    cv2.line(img, points[1], points[2], inference_color, border_thickness)
    cv2.line(img, points[2], points[3], inference_color, border_thickness)
    cv2.line(img, points[3], points[0], inference_color, border_thickness)
    return points


nums_det_more_than_1 = []


# def cal_keypoints_distance_mse(keypoints_ground_truth, keypoints_ground_inference, width, height):
#     kpts1 = [(point[0] / width, point[1] / height) for point in keypoints_ground_truth]
#     kpts2 = [(point[0] / width, point[1] / height) for point in keypoints_ground_inference]
#     return calculate_keypoints_distance_mse(kpts1, kpts2)

def inference_coco_tflite(model_path, input_size, coco_path, image_dir, output_dir, output_csv):
    # Initialize TensorFlow Lite Interpreter.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Model Loaded")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    recheck_dir: str = os.path.join(output_dir, "recheck")
    if not os.path.exists(recheck_dir):
        os.makedirs(recheck_dir)

    print(f"Reading coco from path: {coco_path}")
    coco_data = read_json(coco_path)
    print(
        f"Total images:{len(coco_data['images'])} annotations: {len(coco_data['annotations'])} categories: {len(coco_data['categories'])}")
    inferenceResults = []
    error_image_ids = []
    for image_json in coco_data['images']:
        file_name = image_json['file_name']
        image_path = os.path.join(image_dir, file_name)
        print(f"{image_json['id']}: {image_path}")
        filename = image_path.split("/")[-1]
        img = cv2.imread(image_path)

        if img is not None:
            try:

                original_height, original_width, _ = img.shape
                frame = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                frame_height, frame_width, _ = frame.shape
                frame = frame.astype(np.float32)
                input_tensor = np.expand_dims(frame, axis=0)
                start_time = time.time()



                interpreter.set_tensor(input_details[0]['index'], input_tensor)

                interpreter.invoke()

                scores = interpreter.get_tensor(output_details[2]['index'])  # StatefulPartitionedCall:5
                boxes = interpreter.get_tensor(output_details[0]['index'])  # StatefulPartitionedCall:4
                nums_det = interpreter.get_tensor(output_details[3]['index'])  # StatefulPartitionedCall:3
                classes = interpreter.get_tensor(output_details[5]['index'])  # StatefulPartitionedCall:2
                keypoint = interpreter.get_tensor(output_details[1]['index'])  # StatefulPartitionedCall:1
                keypoint_score = interpreter.get_tensor(output_details[4]['index'])  # StatefulPartitionedCall:0
                keypoints_ground_truth = draw_ground_truth(img, original_width, original_height, image_json,
                                                           coco_data)
                keypoints_ground_inference = draw_inference(img, keypoint[0][0], keypoint_score[0], original_height,
                                                            original_width)

                result = InferenceResult()
                result.image_id = image_json['id']
                result.file_name = file_name
                result.width = original_width
                result.height = original_height
                result.iou_percentage = 100 * calculate_polygon_iou(keypoints_ground_truth,
                                                                    keypoints_ground_inference)

                # เราคำนวนค่าห่างของ keypoint แต่ละจุดเทียบกับขนาดภาพ 1000x1000
                ref_size = 1000
                ref_kpts1 = [(point[0] / original_width * ref_size, point[1] / original_height * ref_size) for point
                             in
                             keypoints_ground_truth]
                ref_kpts2 = [(point[0] / original_width * ref_size, point[1] / original_height * ref_size) for point
                             in
                             keypoints_ground_inference]
                result.distance_by_ref_mse = calculate_keypoints_distance_mse(ref_kpts1, ref_kpts2)
                result.distance_by_ref_p1 = calculate_distance_point(ref_kpts1[0], ref_kpts2[0])
                result.distance_by_ref_p2 = calculate_distance_point(ref_kpts1[1], ref_kpts2[1])
                result.distance_by_ref_p3 = calculate_distance_point(ref_kpts1[2], ref_kpts2[2])
                result.distance_by_ref_p4 = calculate_distance_point(ref_kpts1[3], ref_kpts2[3])
                result.score_inference_p1 = keypoint_score[0][0][0]
                result.score_inference_p2 = keypoint_score[0][0][1]
                result.score_inference_p3 = keypoint_score[0][0][2]
                result.score_inference_p4 = keypoint_score[0][0][3]
                result.inference_p1x = keypoints_ground_inference[0][0]
                result.inference_p1y = keypoints_ground_inference[0][1]
                result.inference_p2x = keypoints_ground_inference[1][0]
                result.inference_p2y = keypoints_ground_inference[1][1]
                result.inference_p3x = keypoints_ground_inference[2][0]
                result.inference_p3y = keypoints_ground_inference[2][1]
                result.inference_p4x = keypoints_ground_inference[3][0]
                result.inference_p4y = keypoints_ground_inference[3][1]
                result.ground_truth_p1x = keypoints_ground_truth[0][0]
                result.ground_truth_p1y = keypoints_ground_truth[0][1]
                result.ground_truth_p2x = keypoints_ground_truth[1][0]
                result.ground_truth_p2y = keypoints_ground_truth[1][1]
                result.ground_truth_p3x = keypoints_ground_truth[2][0]
                result.ground_truth_p3y = keypoints_ground_truth[2][1]
                result.ground_truth_p4x = keypoints_ground_truth[3][0]
                result.ground_truth_p4y = keypoints_ground_truth[3][1]
                inferenceResults.append(result)

                if result.iou_percentage < 97:
                    cv2.imwrite(os.path.join(recheck_dir, filename), img)
                else:
                    cv2.imwrite(os.path.join(output_dir, filename), img)

                if int(nums_det[0]) > 1:
                    nums_det_more_than_1.append(image_json)
                print(os.path.join(output_dir, filename))
                print("Time: ", time.time() - start_time)

            except:
                error_image_ids.append(image_json['id'])
        else:
            print("break")
            break

    with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for result in inferenceResults:
            new_row = [result.image_id,
                       result.file_name,
                       result.width,
                       result.height,
                       round(result.iou_percentage, 2),
                       round(result.distance_by_ref_mse, 2),
                       round(result.distance_by_ref_p1, 2),
                       round(result.distance_by_ref_p2, 2),
                       round(result.distance_by_ref_p3, 2),
                       round(result.distance_by_ref_p4, 2),
                       round(result.score_inference_p1, 2),
                       round(result.score_inference_p2, 2),
                       round(result.score_inference_p3, 2),
                       round(result.score_inference_p4, 2),
                       result.inference_p1x,
                       result.inference_p1y,
                       result.inference_p2x,
                       result.inference_p2y,
                       result.inference_p3x,
                       result.inference_p3y,
                       result.inference_p4x,
                       result.inference_p4y,
                       result.ground_truth_p1x,
                       result.ground_truth_p1y,
                       result.ground_truth_p2x,
                       result.ground_truth_p2y,
                       result.ground_truth_p3x,
                       result.ground_truth_p3y,
                       result.ground_truth_p4x,
                       result.ground_truth_p4y]
            writer.writerow(new_row)
    if len(nums_det_more_than_1) > 0:
        for img_json in nums_det_more_than_1:
            print(f'nums_det more than_ 1 image_id: {img_json["id"]}  file_name: {img_json["file_name"]}')
    if len(error_image_ids) > 0:
        raise ValueError(f"Error in img ids: {error_image_ids}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Draw keypoints and bounding boxes on COCO dataset images and save them.')

    parser.add_argument('--coco_json_path', type=str,
                        default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/DocumentDetection.json',
                        help='Path to COCO annotations JSON file.')
    parser.add_argument('--image_dir', type=str,
                        default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/DocumentDetection',
                        help='Path to images dir.')
    parser.add_argument('--input_size', type=int, default=512,
                        help="Input image size")
    # parser.add_argument('--model_path', type=str,
    #                     default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/train/version7/saved_model_lite_max10_512_default/detect.tflite',
    #                     help='Path to COCO annotations JSON file.')
    parser.add_argument('--model_path', type=str,
                        default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/train/version7/saved_model_lite_max10_512/detect.tflite',
                        help='Path to COCO annotations JSON file.')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/nikornlansa/Workspace/ML/ClearScanner/inference/v7_check_DocumentDetection',
                        help='Path to COCO annotations JSON file.')
    parser.add_argument('--output_csv', type=str,
                        default='/Users/nikornlansa/Workspace/ML/ClearScanner/inference/v7_check_DocumentDetection.csv',
                        help='Path to COCO annotations JSON file.')
    # parser.add_argument('--model_path', type=str,
    #                     default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/train/version1/saved_model_500000/saved_model',
    #                     help='Path to model')
    # parser.add_argument('--output_dir', type=str,
    #                     default='/Users/nikornlansa/Workspace/ML/ClearScanner/inference/v1_500000_check_test',
    #                     help='Path to COCO annotations JSON file.')

    args = parser.parse_args()
    inference_coco_tflite(args.model_path, args.input_size, args.coco_json_path, args.image_dir, args.output_dir, args.output_csv)
