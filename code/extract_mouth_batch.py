''' 
extract_mouth_batch.py
    This script will extract mouth crop of every single video inside source directory
    while preserving the overall structure of the source directory content.

Usage:
    python extract_mouth_batch.py [source directory] [pattern] [target directory] [face predictor path]

    pattern: *.avi, *.mpg, etc 

Example:
    python extract_mouth_batch.py evaluation/samples/GRID/ *.mpg TARGET/

    Will make directory TARGET and process everything inside evaluation/samples/GRID/ that match pattern *.mpg.
'''

import os, fnmatch, sys, errno  
from skimage import io
from preprocess.video import Video
import hyperparameters as hp

SOURCE_PATH = sys.argv[1]
SOURCE_EXTS = sys.argv[2]
TARGET_PATH = sys.argv[3]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def count_files_in_directory(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

for filepath in find_files(SOURCE_PATH, SOURCE_EXTS):
    filepath_wo_ext = os.path.basename(os.path.splitext(filepath)[0])
    target_dir = os.path.join(TARGET_PATH, filepath_wo_ext)
    if os.path.exists(target_dir) and count_files_in_directory(target_dir) >= 73:
        print(f"Skipping {target_dir} as it already contains 75 files.")
        continue
    
    print("Processing: {}".format(filepath))
    video = Video('face').from_video(filepath)
    if video.mouth.ndim != 4:
        continue
    
    mkdir_p(target_dir)

    i = 0
    for frame in video.mouth:
        io.imsave(os.path.join(target_dir, "mouth_{0:03d}.png".format(i)), frame)
        i += 1