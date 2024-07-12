import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
#from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# plt.figure(figsize=(30, 20))
# plt.imshow(load_image_into_numpy_array(image_path))
# plt.show()
# print('Done!!!')

def detect_pose(interpreter, input_tensor, include_keypoint=False):
    """Run detection on an input image.

  Args:
    interpreter: tf.lite.Interpreter
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.
    include_keypoint: True if model supports keypoints output. See
      https://cocodataset.org/#keypoints-2020

  Returns:
    A sequence containing the following output tensors:
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are
        1-based, and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category
        indices.
    If include_keypoints is True, the following are also returned:
      keypoints: (optional) a numpy array of shape [N, 17, 2] representing
        the yx-coordinates of the detection 17 COCO human keypoints
        (https://cocodataset.org/#keypoints-2020) in normalized image frame
        (i.e. [0.0, 1.0]).
      keypoint_scores: (optional) a numpy array of shape [N, 17] representing the
        keypoint prediction confidence scores.
  """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

    interpreter.invoke()

    data = interpreter.get_tensor(output_details[0]['index'])  # StatefulPartitionedCall:5
    return data


def draw_image(input_image_path, output_image_path, csv_row):
    image = cv2.imread(input_image_path)
    (h, w) = image.shape[:2]
    thickness = 3  # Line thickness
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    csv_row = [float(i) for i in csv_row]

    x1, y1 = round(csv_row[0] * w), round(csv_row[1] * h)
    x2, y2 = round(csv_row[2] * w), round(csv_row[3] * h)
    x3, y3 = round(csv_row[4] * w), round(csv_row[5] * h)
    x4, y4 = round(csv_row[6] * w), round(csv_row[7] * h)

    # Draw the line
    cv2.line(image, (x1, y1),  (x2, y2), (255, 0, 0), thickness)
    cv2.line(image, (x2, y2), (x3, y3), (0, 255, 0), thickness)
    cv2.line(image, (x3, y3), (x4, y4), (0, 0, 255), thickness)
    cv2.line(image, (x4, y4), (x1, y1), (0, 255, 255), thickness)

    # Save the image with the drawn line
    cv2.imwrite(output_image_path, image)

def detect_corner(interpreter, input_tensor, include_keypoint=False):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[2]['index']) #StatefulPartitionedCall:5
    boxes = interpreter.get_tensor(output_details[0]['index']) #StatefulPartitionedCall:4
    num_detections = interpreter.get_tensor(output_details[3]['index']) #StatefulPartitionedCall:3
    classes = interpreter.get_tensor(output_details[5]['index']) #StatefulPartitionedCall:2

    if include_keypoint:
        kpts = interpreter.get_tensor(output_details[1]['index']) #StatefulPartitionedCall:1
        kpts_scores = interpreter.get_tensor(output_details[4]['index']) #StatefulPartitionedCall:0
        return boxes, classes, scores, num_detections, kpts, kpts_scores
    else:
        return boxes, classes, scores, num_detections


# Utility for visualizing results
def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    keypoints=None,
                    keypoint_scores=None,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    keypoints: (optional) a numpy array of shape [N, 17, 2] representing the
      yx-coordinates of the detection 17 COCO human keypoints
      (https://cocodataset.org/#keypoints-2020) in normalized image frame
      (i.e. [0.0, 1.0]).
    keypoint_scores: (optional) anumpy array of shape [N, 17] representing the
      keypoint prediction confidence scores.
    figsize: size for the figure.
    image_name: a name for the image file.
  """

    keypoint_edges = [(0, 1),
                      (0, 2),
                      (1, 3),
                      (2, 4),
                      (0, 5),
                      (0, 6),
                      (5, 7),
                      (7, 9),
                      (6, 8),
                      (8, 10),
                      (5, 6),
                      (5, 11),
                      (6, 12),
                      (11, 12),
                      (11, 13),
                      (13, 15),
                      (12, 14),
                      (14, 16)]
    image_np_with_annotations = image_np.copy()
    # Only visualize objects that get a score > 0.3.
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #     image_np_with_annotations,
    #     boxes,
    #     classes,
    #     scores,
    #     category_index,
    #     keypoints=keypoints,
    #     keypoint_scores=keypoint_scores,
    #     keypoint_edges=keypoint_edges,
    #     use_normalized_coordinates=True,
    #     min_score_thresh=0.3)
    # if image_name:
    #     plt.imsave(image_name, image_np_with_annotations)
    # else:
    #     return image_np_with_annotations


# Load the TFLite model and allocate tensors.
# model_path = '/Users/nikornlansa/Workspace/ML/Model/centernet_mobilenetv2_fpn_kpts/model.tflite'
# image_path = '/Users/nikornlansa/Workspace/ML/Dataset/val2017/000000013729.jpg'


#genius scan
# model_path = '/Users/nikornlansa/Workspace/ClearScanner/DeCompiler/decompile model/genius_scan_last/documentFinder.tflite'
# image_path = '/Users/nikornlansa/Workspace/ML/ClearScanner/Images-resize/Document/01/IMG_01_01-01657.JPG'

# model_path = '/Users/nikornlansa/Workspace/ML/Model/output_model_lite12_kp_max_detect_10_320x320/detect.tflite'
# image_path = '/Users/nikornlansa/Workspace/ClearScanner/DocumentCornerLocalizeAndroid/app/src/main/assets/IMG_01_01-01657.JPG'

model_path = '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/train/version7/saved_model_lite_max10_512_default/detect.tflite'
image_path = '/Users/nikornlansa/Workspace/ClearScanner/Custom-keypoint-detection/t/IMG_01_01-00167.JPG'



out_image_path = '/Users/nikornlansa/Downloads/out_tflite.JPG'

# Initialize TensorFlow Lite Interpreter.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Keypoints are only relevant for people, so we only care about that
# category Id here.
category_index = {1: {'id': 1, 'name': 'person'}}

label_id_offset = 1
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
height = image.shape[0]
width = image.shape[1]

print("Height:",height)
print("Width:",width)
image = tf.expand_dims(image, axis=0)
image_numpy = image.numpy()

input_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.float32)
# Note that CenterNet doesn't require any pre-processing except resizing to
# input size that the TensorFlow Lite Interpreter was generated with.
# input_tensor = tf.image.resize(input_tensor, (320, 320))
# (boxes, classes, scores, num_detections, kpts, kpts_scores) = detect_pose(
#     interpreter, input_tensor, include_keypoint=True)

input_tensor = tf.image.resize(input_tensor, (512, 512))
(boxes, classes, scores, num_detections, kpts, kpts_scores) = detect_corner(
    interpreter, input_tensor, include_keypoint=True)
keypoints = kpts[0][0]
csv_row = [keypoints[0][1], keypoints[0][0], keypoints[1][1], keypoints[1][0], keypoints[2][1], keypoints[2][0], keypoints[3][1], keypoints[3][0]]
draw_image(image_path, out_image_path, csv_row)

# vis_image = plot_detections(
#     image_numpy[0],
#     boxes[0],
#     classes[0].astype(np.uint32) + label_id_offset,
#     scores[0],
#     category_index,
#     keypoints=kpts[0],
#     keypoint_scores=kpts_scores[0])
# plt.figure(figsize=(30, 20))
# plt.imshow(vis_image)
