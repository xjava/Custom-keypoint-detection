
import cv2
import numpy as np


def get_document_corners(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Display edged image for debugging
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)

    # Detect lines using HoughLines
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 150)

    if lines is None:
        raise ValueError("No lines detected.")

    # Find the intersection points of the lines
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    points = []
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        points.append(((x1, y1), (x2, y2)))

    intersections = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            intersection = line_intersection(points[i], points[j])
            if intersection:
                intersections.append(intersection)

    # Filter points that lie within the image boundaries
    intersections = [pt for pt in intersections if 0 <= pt[0] <= image.shape[1] and 0 <= pt[1] <= image.shape[0]]

    # Display intersections for debugging
    debug_image = image.copy()
    for pt in intersections:
        cv2.circle(debug_image, tuple(map(int, pt)), 5, (0, 0, 255), -1)

    cv2.imwrite("/Users/nikornlansa/Downloads/out_autodetect_debug.jpg", debug_image)

    # Apply clustering to group intersection points and find the corners
    def cluster_points(points, threshold=10):
        clusters = []
        for point in points:
            for cluster in clusters:
                if np.linalg.norm(np.array(point) - np.array(cluster)) < threshold:
                    cluster[0] = (cluster[0] * cluster[1] + point[0]) / (cluster[1] + 1)
                    cluster[1] += 1
                    break
            else:
                clusters.append([point[0], 1])
        return [cluster[0] for cluster in clusters]

    corners = cluster_points(intersections)

    if len(corners) < 4:
        raise ValueError("Not enough corners detected. Make sure the document is clearly visible and well-lit.")

    # Sort corners to get top-left, top-right, bottom-right, and bottom-left
    corners = np.array(corners)
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    top_left = corners[np.argmin(s)]
    bottom_right = corners[np.argmax(s)]
    top_right = corners[np.argmin(diff)]
    bottom_left = corners[np.argmax(diff)]

    # Draw the corners on the image
    for corner in [top_left, top_right, bottom_right, bottom_left]:
        cv2.circle(image, tuple(map(int, corner)), 10, (0, 255, 0), -1)

    # Draw the document borders
    cv2.line(image, tuple(map(int, top_left)), tuple(map(int, top_right)), (255, 0, 0), 2)
    cv2.line(image, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (255, 0, 0), 2)
    cv2.line(image, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (255, 0, 0), 2)
    cv2.line(image, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (255, 0, 0), 2)

    # Display the result
    cv2.imshow("Document Corners and Borders", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return the corner points
    return top_left, top_right, bottom_right, bottom_left

    # Display the result
    # cv2.imshow("Document Corners and Borders", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("/Users/nikornlansa/Downloads/out_autodetect.jpg", image)

    # Return the corner points
    return top_left, top_right, bottom_right, bottom_left


# Example usage
image_path = '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/CardDetection/IMG_03_01-00001.JPG'
#image_path =  '/Users/nikornlansa/Downloads/8623CEA0-6CDA-4C0E-A08F-67E642E5F209.JPG'
corners = get_document_corners(image_path)
print("Document corners:", corners)
