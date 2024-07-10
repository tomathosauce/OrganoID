
# Define the source and destination directories
from pathlib import Path
import shutil


source_dir = Path(r'SURE\datasets\original\OriginalData\training\pre_augmented')
source_dir_img = source_dir / 'images'
source_dir_seg = source_dir / 'segmentations'

destination_dir = Path(r'SURE\datasets\handpicked\3\pre_augmented')
destination_dir_img = destination_dir / 'original'
destination_dir_seg = destination_dir / 'segmentations'

# Ensure the destination directory exists
destination_dir.mkdir(parents=True, exist_ok=True)

# List of files to copy
files_to_copy = [
    32,33,46,46,47,48,50,51,52,55,59,17,15,16,26
]

def copy_file(source_dir: Path, destination_dir: Path, file_name: str):
    source_file = source_dir / file_name
    destination_file = destination_dir / file_name
    if source_file.exists():
        shutil.copy(source_file, destination_file)
        print(f'Copied {source_file} to {destination_file}')
    else:
        print(f'File {source_file} does not exist')

# Iterate through the list of files and copy each one to the destination directory
for file_name in files_to_copy:
    fn = str(file_name) + '.png'
    # copy_file(source_dir=source_dir_img, destination_dir=destination_dir_img, file_name=fn)
    copy_file(source_dir=source_dir_seg, destination_dir=destination_dir_seg, file_name=fn)
    