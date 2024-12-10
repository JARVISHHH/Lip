import os
from pathlib import Path

#helper functoins for data preprocessing 
def remove_file(out_dir):
    for subdir in out_dir.iterdir():
        if subdir.is_dir():  
            for file in subdir.glob('*.align'):
                print(f"Removing file: {file}")  
                file.unlink()  

#Check which video folder missing from preprocessing
def compare_folders(in_dir_path, out_dir_path):
    in_dir = Path(in_dir_path)
    out_dir = Path(out_dir_path)
    missed_files = {}
    
    for in_subdir in in_dir.glob('**/*'):
        if in_subdir.is_file() and in_subdir.suffix == '.mpg':
            # Remove the extension from the file name to match against folder names
            expected_dir_name = in_subdir.stem
            # Construct the path to the expected directory in out_dir
            expected_dir_path = out_dir / in_subdir.parent.relative_to(in_dir) / expected_dir_name
       
            if not expected_dir_path.exists():
                relative_path = in_subdir.parent.relative_to(in_dir)
                if relative_path in missed_files:
                    missed_files[relative_path] += 1
                else:
                    missed_files[relative_path] = 1
    print(missed_files)
            
in_dir_path = "../datasets/videos"
out_dir_path = "../datasets/word_alignments"
# compare_folders(in_dir_path, out_dir_path)

    