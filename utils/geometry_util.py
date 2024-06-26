import numpy as np
import math
from shapely.geometry import box, Polygon


def get_polygon_area(x, y):
    """https://en.wikipedia.org/wiki/Shoelace_formula"""

    assert len(x) == len(y)

    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_polygon_bounding_box(x, y):
    assert len(x) == len(y)

    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
    return [x1, y1, x2 - x1, y2 - y1]


def calculate_polygon_iou(pol1_xy, pol2_xy):  # https://stackoverflow.com/a/64358582
    # Define each polygon
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    # Calculate intersection and union, and tne IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union

# pl1 = [[95.238097012043, 118.22660267353058], [894.909679889679, 129.31033968925476], [868.6370849609375, 940.8866763114929], [113.30049484968185, 947.044312953949]]
# pl2 = [[496.78581953048706, 496.4821934700012], [516.460657119751, 472.3222851753235], [505.057692527771, 478.86818647384644], [491.6919469833374, 508.5298418998718]]
# print(calculate_polygon_iou(pl1, pl2))

def calculate_distance_point(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
# Example usage:
# point1 = (1, 2)
# point2 = (4, 6)
# distance = calculate_distance(point1[0], point1[1], point2[0], point2[1])
# print(f"The distance between points {point1} and {point2} is {distance}")


# ตัวอย่างโค้ด chatgpt Compute the MSE: Use the Euclidean distance for each corner and then average the distances.
def calculate_keypoints_distance_mse(ground_truth, predicted):
    return np.mean([np.sum((np.array(gt) - np.array(pred)) ** 2) for gt, pred in zip(ground_truth, predicted)])

# ground_truth = [(x1_gt, y1_gt), (x2_gt, y2_gt), (x3_gt, y3_gt), (x4_gt, y4_gt)]
# predicted = [(x1_pred, y1_pred), (x2_pred, y2_pred), (x3_pred, y3_pred), (x4_pred, y4_pred)]
# error = mse(ground_truth, predicted)