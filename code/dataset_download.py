import gdown
import os
import glob

url = 'https://drive.google.com/drive/u/1/folders/1nZ1Vg_fBOYAiu_cxt762WLC-A1YKBz5a'
output = "../".replace("\\", os.sep).replace("/", os.sep)
gdown.download_folder(url, output=output)

datasets_path = os.path.join(output, "datasets")

# video
mouth_path = os.path.join(datasets_path, "mouth")
frames_zip_path = os.path.join(mouth_path, "*", "*.zip")
for frames_path in glob.glob(frames_zip_path):
    print("Extracting {}...".format(frames_path))
    gdown.extractall(frames_path)
    os.remove(frames_path)

# alignments
alignments_zip_path = os.path.join(datasets_path, "alignments", "alignments.zip")
print("Extracting {}...".format(alignments_zip_path))
gdown.extractall(alignments_zip_path)
os.remove(alignments_zip_path)