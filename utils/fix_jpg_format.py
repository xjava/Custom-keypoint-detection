from PIL import Image
import os
import cv2

#เอาไว้แก้ไฟล์นามสกุล .JPG ที่ไม่ใช่ JPG ไฟล์
def convert_to_jpg(image_path):
    try:
        # print(f"{image_path}...")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
    except Exception as e:
        print(f"{image_path} is no JPG format converting...")
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path, 'JPEG')


def fix_images_in_directory(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                image_path = os.path.join(root, filename)
                convert_to_jpg(image_path)
            elif filename.endswith('DS_Store'):
                os.remove(os.path.join(root, filename))
            else:
                raise TypeError("Unknown file extension {}".format(
                    filename))  # ก็เพิ่มไปใน if ข้างบน เขียนกันเผื่อเจอไฟล์แปลกๆแค่นั้น


def main():
    current_directory = '/Users/nikornlansa/Downloads/image-clean'
    fix_images_in_directory(current_directory)


if __name__ == "__main__":
    main()
    print("done")
