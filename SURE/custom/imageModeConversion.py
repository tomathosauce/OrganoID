import os
import pickle
from PIL import Image, ImageFilter
import numpy as np

import argparse

def handle_i16(input_path, output_path, mode):
    # Load image
    ims = pickle.load(open(input_path, "rb"))

    # Normalise to range 0..255
    norm = (ims.astype(np.float)-ims.min())*255.0 / (ims.max()-ims.min())

    # Save as 8-bit PNG
    img = Image.fromarray(norm.astype(np.uint8))
    
    # converted_img = img.convert(mode)
    # converted_img.save(output_path)
    img.save(output_path)

def convert_images_in_directory(input_dir, output_dir, target_mode='RGBA'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List all files in the input directory
    files = os.listdir(input_dir)
    
    # Filter out non-image files (optional)
    image_files = [file for file in files if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))]
    
    # Process each image file
    for image_file in image_files:
        # Construct full file paths
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        try:
            # Open image
            with Image.open(input_path) as img:
                # Print the original mode
                print(f"Processing {image_file}: Original mode: {img.mode}")
                if img.mode == ['I;16']:
                    # handle_i16(input_path, output_path, target_mode)
                    # print(f"Converted mode: {converted_img.mode}")# Convert from I;16 to 8-bit grayscale

                    # Convert to 8-bit grayscale
                    converted_img = img.point(lambda i:i*(1./256)).convert('L')#.filter(ImageFilter.BLUR)
                elif img.mode in ['L', 'I']:
                    converted_img = img
                else:
                    # Convert image to the target mode
                    converted_img = img.convert(target_mode)
                
                # Print the converted mode
                print(f"Converted mode: {converted_img.mode}")
                
                # Save the converted image
                converted_img.save(output_path)
                print(f"Converted image saved to: {output_path}")
        
        except IOError:
            print(f"Cannot process {image_file}")
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="OrganoID image preprocessor")

    parser.add_argument('input_path', type=str, help='Path of an image or directory containing images')
    parser.add_argument('output_path', type=str, help='Path of output directory for images')
    parser.add_argument('-m', '--mode', type=str, help='Target image mode', default='RGBA')

    args = parser.parse_args()

    convert_images_in_directory(args.input_path, args.output_path, args.mode)