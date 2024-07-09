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
    edged = cv2.Canny(blurred, 50, 200)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and get the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    document_contour = None

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the approximated contour has four points, it is likely the document
        if len(approx) == 4:
            document_contour = approx
            break

    if document_contour is None:
        raise ValueError("Document contour not found.")

    # Draw the contour on the image
    cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 2)

    # Display the result
    cv2.imwrite("/Users/nikornlansa/Downloads/out_autodetect.jpg", image)

    # Return the corner points
    return document_contour.reshape(4, 2)


# Example usage
#image_path = '/Users/nikornlansa/Workspace/ML/ClearScanner/Images-resize/Document/01/IMG_01_01-00016.JPG'
image_path =  '/Users/nikornlansa/Workspace/ML/ClearScanner/sync/DocumentCornerLocalization/datasets/CardDetection/IMG_03_01-00001.JPG'
corners = get_document_corners(image_path)
print("Document corners:", corners)

