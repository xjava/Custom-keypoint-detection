import os
import argparse
import cv2
from pycocotools.coco import COCO

def draw_keypoints_and_bboxes_on_images(coco_json_path, output_folder):
    parent = os.path.dirname(coco_json_path)
    filename, ext = os.path.splitext(os.path.basename(coco_json_path))

    images_folder = os.path.join(parent, filename)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load COCO annotations
    coco = COCO(coco_json_path)

    # Loop through all images in the COCO dataset
    for image_info in coco.loadImgs(coco.getImgIds()):
        image_path = os.path.join(images_folder, image_info['file_name'])
        image = cv2.imread(image_path)
        # Get original dimensions
        (h, w) = image.shape[:2]

        fontScale = 2
        adjust = 60
        circle_size = 8
        border_thickness = 2
        if h < 750 and w < 750:
            fontScale = 1
            adjust = 30
            circle_size = 4
            border_thickness = 1

        if image is None:
            print(f"Error loading image {image_path}")
            continue

        # Get all annotations for the current image
        annotation_ids = coco.getAnnIds(imgIds=image_info['id'], iscrowd=False)
        annotations = coco.loadAnns(annotation_ids)
        print(f"Drawing {image_info['file_name']}")
        for annotation in annotations:
            # Draw bounding box
            if 'bbox' in annotation:
                x, y, width, height = annotation['bbox']
                x, y, width, height = int(x), int(y), int(width), int(height)
                cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

            # Draw keypoints
            points = []
            if 'keypoints' in annotation:
                keypoints = annotation['keypoints']
                index = 0
                for i in range(0, len(keypoints), 3):
                    kp_x = keypoints[i]
                    kp_y = keypoints[i + 1]
                    v = keypoints[i + 2]  # Visibility flag
                    if v > 0:  # Draw keypoint if visible
                        cv2.circle(image, (kp_x, kp_y), circle_size, (0, 0, 255), -1)
                        points.append((kp_x, kp_y))

                        if index == 0 or index == 1:
                            kp_y = kp_y + adjust
                        else:
                            kp_y = kp_y - adjust

                        if index == 0 or index == 3:
                            kp_x = kp_x + adjust
                        else:
                            kp_x = kp_x - adjust
                        cv2.putText(
                            image,  # image on which to draw text
                            f"{index + 1}",
                            (kp_x, kp_y),  # bottom left corner of text
                            cv2.FONT_HERSHEY_SIMPLEX,  # font to use
                            fontScale,  # font scale
                            (0, 255, 0),  # color
                            5,  # line thickness
                        )
                        index = index + 1
                # Draw the line
            # border_color = (0, 255, 0)
            # cv2.line(image, points[0], points[1], border_color, border_thickness)
            # cv2.line(image, points[1], points[2], border_color, border_thickness)
            # cv2.line(image, points[2], points[3], border_color, border_thickness)
            # cv2.line(image, points[3], points[0], border_color, border_thickness)
            # Draw Image ID
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # bottomLeftCornerOfText = (10, 500)
            # fontScale = 1
            # fontColor = (0, 255, 0)
            # thickness = 1
            # lineType = 2
            #
            # cv2.putText(image, 'Hello World!',
            #             bottomLeftCornerOfText,
            #             font,
            #             fontScale,
            #             fontColor,
            #             thickness,
            #             lineType)
            cv2.putText(
                image,  # image on which to draw text
                f"ID : {image_info['id']}",
                (int(w / 2) - 120, adjust),  # bottom left corner of text
                cv2.FONT_HERSHEY_SIMPLEX,  # font to use
                fontScale,  # font scale
                (0, 255, 0),  # color
                5,  # line thickness
            )


        # Save the image with keypoints and bounding boxes drawn
        output_path = os.path.join(output_folder, image_info['file_name'])
        cv2.imwrite(output_path, image)

    print("Keypoints and bounding boxes have been drawn and saved to the output folder.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw keypoints and bounding boxes on COCO dataset images and save them.')
    parser.add_argument('--coco_json_path', type=str, default='/Users/nikornlansa/Workspace/ML/ClearScanner/Images-resize/Document/01.json', help='Path to COCO annotations JSON file.')
    parser.add_argument('--output_dir', type=str, default='/Users/nikornlansa/Downloads/DocumentDetection3_draw',
                        help='Path to COCO annotations JSON file.')

    args = parser.parse_args()
    draw_keypoints_and_bboxes_on_images(args.coco_json_path, args.output_dir)
