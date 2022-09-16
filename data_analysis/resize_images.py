import os
from PIL import Image


# Resize an image to a user-defined maximum width/height with the same aspect ratio
def resize_images(folder_dir, maxlength):
    for subdir, dirs, files in os.walk(folder_dir):
        for file in files:
            if not str(os.path.join(file)).startswith('.'): # Ignore the hidden file
                filepath = os.path.join(subdir, file)
                print(filepath)
                img = Image.open(filepath)
                height, width = img.size
                print([height, width])

                if height >= width:
                    size_ratio = (maxlength/float(height))
                    width = int((float(width)*float(size_ratio)))
                    img = img.resize((maxlength,width), Image.ANTIALIAS)
                    img.save(filepath, 'JPEG')

                if width >= height:
                    size_ratio = (maxlength/float(width))
                    height = int((float(height)*float(size_ratio)))
                    img = img.resize((height, maxlength), Image.ANTIALIAS)
                    img.save(filepath, 'JPEG')


if __name__ == '__main__':
    # Set the path to Wikiart images
    # folder_dir = '/Users/Cherry0904/Desktop/wikiart_test'
    folder_dir = '/content/drive/MyDrive/Github/wikiart'
    maxlength = 600

    Image.MAX_IMAGE_PIXELS = None
    resize_images(folder_dir, maxlength)