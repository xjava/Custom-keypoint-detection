import docdetect
import cv2

image_path = '/Users/nikornlansa/Workspace/ML/ClearScanner/Images-resize/Document/01/IMG_01_01-00016.JPG'
image = cv2.imread(image_path)
rects = docdetect.process(image)
image = docdetect.draw(rects, image)
cv2.imwrite("/Users/nikornlansa/Downloads/out_autodetect.jpg", image)