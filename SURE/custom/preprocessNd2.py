import os
import pprint
import re
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pims
from PIL import Image, ImageOps
import argparse

parser = argparse.ArgumentParser(description="OrganoID image preprocessor")

parser.add_argument('-n', '--normalize', action='store_true', help='Normalize images')
parser.add_argument('input_path', type=str, help='Path of an image or directory containing images')
parser.add_argument('output_path', type=str, help='Path of output directory for images')
parser.add_argument('-p', '--padding', action='store_true', help='Pad the image(s)')

args = parser.parse_args()

def get_avg_pixel(img):
    # Resize the image to 1x1 pixel
    small_image = img.resize((1, 1))

    # Get the color of the single pixel
    average_color = small_image.getpixel((0, 0))
    
    return average_color

def pad_image(img, padding=100):
    avg_color = get_avg_pixel(img)
    # Create a new image with the desired padding
    return ImageOps.expand(img, border=padding, fill=avg_color)  # Change 'black' to your desired color

def normalize_image(img):
    img_array = np.array(img)

    # Get the min and max values
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    print(img_array.shape)
    # Normalize to [0, 255]
    normalized_array = (img_array - min_val) / (max_val - min_val)
    normalized_array *= 255
    normalized_array = normalized_array.astype(np.uint8)

    # Convert back to Pillow Image
    normalized_img = Image.fromarray(normalized_array)
    return normalized_img

def convert_nd2_to_png(input_path, output_path, reader=pims.Bioformats):
    """
    Converts ND2 images to PNG format with auto-adjusted colormap.
    
    Parameters:
        input_path (str): Path to the input ND2 file.
        output_path (str): Directory where PNG files will be saved.
    """
    # Check if the output directory exists, create if it doesn't
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    if os.path.isfile(input_path):
        with reader(input_path) as images:
            plt.imshow(images[0])
    else:
        f = []
        for (dirpath, dirnames, filenames) in os.walk(input_path):
            f.extend(filenames)
            break
        
        for fn in f:
            try:
                with reader(f"{input_path}\\{fn}") as images:
                    # print('%d x %d px' % (images.metadata['width'], images.metadata['height']))
                    # pprint.pprint(images.metadata)
                    meta = images.metadata
                    image_count = meta.ImageCount()
                    print('Total number of images: {}'.format(image_count))

                    for i, image in enumerate(images):
                        print('Dimensions for image {}'.format(i))
                        shape = (meta.PixelsSizeX(i), meta.PixelsSizeY(i), meta.PixelsSizeZ(i))
                        print('\tShape: {} x {} x {}'.format(*shape))
                        
                        # normalized_image = normalize_image(image)
                        # Create an Image object from the array
                        img = Image.fromarray(image)
                        
                        if args.normalize:
                            img = normalize_image(img)
                        
                        if args.padding:
                            width, height = img.size
                            pad = max(width, height) // 2
                            img = pad_image(img, pad)
                        # Save the image as PNG
                        if image_count == 1:
                            img.save(os.path.join(output_path, f'{fn}.png'))
                        elif image_count > 1:
                            pattern = r'([A-Z])\w+'
                            search = re.search(pattern, fn)
                            
                            if search:
                                img_dir_path = os.path.join(output_path, search.group())
                                if not os.path.exists(img_dir_path):
                                    os.makedirs(img_dir_path)
                                img.save(os.path.join(img_dir_path, f'{fn}_{i}.png'))
                            else:
                                raise Exception("Invalid Path name")
                            
                    # pprint.pprint(images.sizes)
            except Exception as e:
                print(f"Error with {fn}")
                print(e)
                traceback.print_exc()

# Example usage
input_nd2_file = args.input_path# 'Dataset'
output_directory = args.output_path #'Postprocessed'

convert_nd2_to_png(input_nd2_file, output_directory)
