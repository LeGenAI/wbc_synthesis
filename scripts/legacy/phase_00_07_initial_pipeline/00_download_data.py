"""
Step 0: Download Kaggle Peripheral Blood Cell dataset.
Run this first.
"""
import kagglehub
import shutil
import os

DEST = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(DEST, exist_ok=True)

print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("bzhbzh35/peripheral-blood-cell")
print(f"Downloaded to: {path}")

# Copy/link into project raw dir
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(DEST, item)
    if not os.path.exists(dst):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

print(f"Data available at: {os.path.abspath(DEST)}")
print("Contents:", os.listdir(DEST))
