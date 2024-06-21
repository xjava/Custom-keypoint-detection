import numpy as np
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


# ตัวอย่างโค้ด chatgpt Compute the MSE: Use the Euclidean distance for each corner and then average the distances.
def mse(ground_truth, predicted):
    return np.mean([np.sum((np.array(gt) - np.array(pred)) ** 2) for gt, pred in zip(ground_truth, predicted)])

# ground_truth = [(x1_gt, y1_gt), (x2_gt, y2_gt), (x3_gt, y3_gt), (x4_gt, y4_gt)]
# predicted = [(x1_pred, y1_pred), (x2_pred, y2_pred), (x3_pred, y3_pred), (x4_pred, y4_pred)]
# error = mse(ground_truth, predicted)