import gdown
import os
import glob
import argparse
import shutil

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="download and extract datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--skip-download',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='''skip download.''')
    parser.add_argument(
        '--skip-extract',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='''skip extract.''')

    return parser.parse_args()

ARGS = parse_args()

url = 'https://drive.google.com/drive/u/1/folders/1nZ1Vg_fBOYAiu_cxt762WLC-A1YKBz5a'
output = "../".replace("\\", os.sep).replace("/", os.sep)
downloads_path = os.path.join(output, "downloads")

if not ARGS.skip_download:
    gdown.download_folder(url, output=downloads_path)

datasets_path = os.path.join(output, "datasets")

# video
mouth_download_path = os.path.join(downloads_path, "mouth")
mouth_extract_path = os.path.join(datasets_path, "mouth")
frames_zip_path = os.path.join(mouth_download_path, "*", "*.zip")
if not ARGS.skip_extract:
    for frames_path in glob.glob(frames_zip_path):
        train_val = frames_path.split(os.sep)[-2]
        folder_name = os.path.splitext(frames_path.split(os.sep)[-1])[0]
        if int(folder_name[1:]) < 18:
            extract_path = os.path.join(mouth_extract_path, train_val)
            print("Extracting {} to {}...".format(frames_path, extract_path))
            gdown.extractall(frames_path, extract_path)
        elif int(folder_name[1:]) >= 24 and int(folder_name[1:]) <= 29 or int(folder_name[1:]) >= 32 and int(folder_name[1:]) <= 34:
            extract_path = os.path.join(mouth_extract_path, train_val)
            print("Extracting {} to {}...".format(frames_path, extract_path))
            gdown.extractall(frames_path, extract_path)
            shutil.rmtree(os.path.join(extract_path, "__MACOSX"))
        else:
            extract_path = os.path.join(mouth_extract_path, train_val, folder_name)
            print("Extracting {} to {}...".format(frames_path, extract_path))
            gdown.extractall(frames_path, extract_path)

# alignments
alignments_zip_path = os.path.join(downloads_path, "alignments", "alignments.zip")
extract_path = os.path.join(datasets_path, "alignments")
if not ARGS.skip_extract:
    print("Extracting {} to {}...".format(alignments_zip_path, extract_path))
    gdown.extractall(alignments_zip_path, extract_path)