from pathlib import Path

def empty_directory(directory_path):
    directory = Path(directory_path)
    
    # Ensure the directory exists
    if not directory.is_dir():
        raise ValueError(f"The path {directory_path} is not a directory or does not exist.")
    
    # Iterate over all items in the directory
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()  # Remove the file
        elif item.is_dir():
            # Recursively delete the subdirectory
            empty_directory(item)
            item.rmdir()  # Remove the now-empty subdirectory