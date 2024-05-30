from PIL import Image
import os
#run ผ่าน command line เพราะใน pyvharm install pillow_heif ไม่ได้ python convert_to_jpg.py
from pillow_heif import register_heif_opener

register_heif_opener()
def convert_to_jpg(image_path):
    try:
        image = Image.open(image_path)

        if image.mode != 'RGB':
            print(image_path)
            image = image.convert('RGB')

        # Get the filename without extension
        filename, ext = os.path.splitext(image_path)

        if ext.lower() == '.jpg':
            # Save the existing jpg file with lower case extension
            if(ext != '.jpg'):
                os.rename(image_path, filename + ext.lower())
                print(f"{image_path} renamed to {filename + ext.lower()}.")

        else:
            # Save as jpg
            image.save(filename + '.jpg', 'JPEG')
            os.remove(image_path)
            print(f"{image_path} converted to JPEG. And Remove!")
    except Exception as e:
        print(f"Error converting {image_path}: {e}")


def convert_images_in_directory(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpeg', '.jpg', '.bmp', '.gif', '.tiff', '.heic', '.webp')):
                image_path = os.path.join(root, filename)
                convert_to_jpg(image_path)
            elif filename.endswith('DS_Store'):
                os.remove(os.path.join(root, filename))
            else:
                raise TypeError("Unknown file extension {}".format(filename)) #ก็เพิ่มไปใน if ข้างบน เขียนกันเผื่อเจอไฟล์แปลกๆแค่นั้น


def main():
    current_directory = '/Users/nikornlansa/Downloads/image_v2/doc/test5'
    convert_images_in_directory(current_directory)


if __name__ == "__main__":
    main()
    print("done")
