#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:59:00 2022

@author: prabhakar
"""


import json
import os
import funcy
import argparse
from sklearn.model_selection import train_test_split
from coco_dataset_util import read_coco, write_coco

def save_coco(file, info, licenses, images, annotations, categories):
    coco_data = { 'info': info, 'licenses': licenses, 'images': images,
            'annotations': annotations, 'categories': categories}
    write_coco(file, coco_data)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def split_coco(coco_path):
    parent = os.path.dirname(coco_path)
    filename, ext = os.path.splitext(os.path.basename(coco_path))
    train_path = os.path.join(parent, f"{filename}_train.json")
    val_path = os.path.join(parent, f"{filename}_val.json")
    test_path = os.path.join(parent, f"{filename}_test.json")

    coco = read_coco(coco_path)
    info = coco['info'] if 'info' in coco else {}
    licenses = coco['licenses'] if 'licenses' in coco else {}
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
    images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

    train, val_test = train_test_split(images, train_size=0.75)#แบ่ง train 75%, val 15%, test 10%
    val, test = train_test_split(val_test, train_size=0.60)#ที่เหลือเอามาแบ่งอีกทีเพื่อให้ได้ val 15%, test 10% ของทั้งหมด

    save_coco(train_path, info, licenses, train, filter_annotations(annotations, train), categories)
    save_coco(val_path, info, licenses, val, filter_annotations(annotations, val), categories)
    save_coco(test_path, info, licenses, test, filter_annotations(annotations, test), categories)


if __name__ == "__main__":
    split_coco("/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/dataset/DocumentDetection.json")
    split_coco("/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/dataset/DocumentDetection2.json")
    split_coco("/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/dataset/BoardDetection.json")
    split_coco("/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/dataset/CardDetection.json")
    split_coco("/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/dataset/BookDetection.json")
    # split_coco("/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/dataset/MendeleyDetection.json")
    # split_coco("/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalization/dataset/CordDetection.json")
    #ปกติเราจะ split dataset แต่ละตัวครั้งเดียวและจะเก็บไว้แบบนี้เพื่อใช้ train และ test