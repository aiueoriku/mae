import os
from PIL import Image

folder = "../cryoppp_denoised/10005"  # チェックしたいフォルダのパスに変更してください

for filename in os.listdir(folder):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        filepath = os.path.join(folder, filename)
        try:
            with Image.open(filepath) as img:
                img.verify()  # 画像が壊れていないか検証
            print(f"OK: {filename}")
        except Exception as e:
            print(f"NG: {filename} ({e})")