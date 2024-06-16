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
def list_files(directory, extensions):
    # List all files in the given directory
    files = os.listdir(directory)
    # Filter out the files that end with .csv

    filtered_files = [file for file in files if file.endswith(extensions)]
    return filtered_files


colors = [(0, 0, 255), (0, 255, 0), (255, 0, 255),
          (255, 0, 0)]


# cv2.namedWindow("display", cv2.WINDOW_NORMAL)

def process_keypoint(kp, kp_s, h, w, img):
    for i, kp_data in enumerate(kp):
        cv2.circle(img, (int(kp_data[1] * w), int(kp_data[0] * h)), 4, colors[i], -1)
    return img


def inference_coco(model_path, coco_path, image_dir, output_dir):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.load(sess, ['serve'], model_path)
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name("serving_default_input_tensor:0")
        det_score = graph.get_tensor_by_name("StatefulPartitionedCall:6")
        det_class = graph.get_tensor_by_name("StatefulPartitionedCall:2")
        det_boxes = graph.get_tensor_by_name("StatefulPartitionedCall:0")
        det_numbs = graph.get_tensor_by_name("StatefulPartitionedCall:7")
        det_keypoint = graph.get_tensor_by_name("StatefulPartitionedCall:4")
        det_keypoint_score = graph.get_tensor_by_name("StatefulPartitionedCall:3")
        print("Model Loaded")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Reading coco from path: {coco_path}")
        coco_data = read_json(coco_path)
        print(
            f"Total images:{len(coco_data['images'])} annotations: {len(coco_data['annotations'])} categories: {len(coco_data['categories'])}")

        files = [image['file_name'] for image in coco_data['images']]
        files.sort()
        for file in files:
            image_path = os.path.join(image_dir, file)
            print(image_path)
            filename = image_path.split("/")[-1]
            frame = cv2.imread(image_path)
            if frame is not None:
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
                height, width, _ = frame.shape
                image_exp_dims = np.expand_dims(frame, axis=0)
                start_time = time.time()
                score, classes, boxes, nums_det, keypoint, keypoint_score = sess.run([det_score, det_class, det_boxes,
                                                         det_numbs, det_keypoint, det_keypoint_score],
                                                        feed_dict={input_tensor: image_exp_dims})
                for i in range(int(nums_det[0])):
                    if (score[0][i] * 100 > 50):
                        per_box = boxes[0][i]
                        y1 = int(per_box[0] * height)
                        x1 = int(per_box[1] * width)
                        y2 = int(per_box[2] * height)
                        x2 = int(per_box[3] * width)

                        p1 = (x1, y1)
                        p2 = (x2, y2)
                        cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
                frame = process_keypoint(keypoint[0][0], keypoint_score[0], height, width, frame)
                #            cv2.imshow("display",frame)
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                print(os.path.join(output_dir, filename))
                print("Time: ", time.time() - start_time)
            #            cv2.waitKey(1)
            else:
                print("break")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Draw keypoints and bounding boxes on COCO dataset images and save them.')

    parser.add_argument('--coco_json_path', type=str,
                        default='/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/train_config/version_1/dataset/train_data.json',
                        help='Path to COCO annotations JSON file.')
    parser.add_argument('--image_dir', type=str,
                        default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/images',
                        help='Path to images dir.')



    # parser.add_argument('--model_path', type=str,
    #                     default='/Users/nikornlansa/Workspace/ML/Model/results/saved_model12/saved_model',
    #                     help='Path to COCO annotations JSON file.')
    # parser.add_argument('--output_dir', type=str,
    #                     default='/Users/nikornlansa/Workspace/ML/ClearScanner/inference/v12_check_train',
    #                     help='Path to COCO annotations JSON file.')
    parser.add_argument('--model_path', type=str,
                        default='/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/train/version1/saved_model_200000/saved_model',
                        help='Path to model')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/nikornlansa/Workspace/ML/ClearScanner/inference/v1_200000_check_train',
                        help='Path to COCO annotations JSON file.')

    args = parser.parse_args()
    inference_coco(args.model_path, args.coco_json_path, args.image_dir, args.output_dir)
