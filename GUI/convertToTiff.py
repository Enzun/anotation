# Predict Phase
# たくさん画像いれてgui.pyで良くないやつをピックアップしてneedFixへ
# convertToTiff.pyでneedFixからTiff画像にしてTiffsに保存。
# Tiffにしたらノートパソコンから取り出してCVATにあげる ->Annotation Phase。
# -> Training Phase

#GUI/convertTotiff.py

import os
import pydicom
from PIL import Image
import numpy as np

# Tiffファイルを保存するための出力ディレクトリ
output_dir = 'Tiffs/'
os.makedirs(output_dir, exist_ok=True)

# DICOMファイルが保存されているディレクトリ
dicom_dir = 'needFix'

# ディレクトリ内のすべてのファイルを取得し、DICOMファイルのみをリストに追加
dicom_files = [f for f in os.listdir(dicom_dir) if f.startswith('EX')]

# DICOMファイルをループ処理してtiff化
for i in range(len(dicom_files)):
    file_name = str(dicom_files[i])
    file_path = os.path.join(dicom_dir, file_name)
    ds = pydicom.dcmread(file_path)
    
    # 画像データを取得
    image_array = ds.pixel_array
    
    # ウィンドウ幅とウィンドウ中心を取得
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        window_center = ds.WindowCenter if isinstance(ds.WindowCenter, float) else ds.WindowCenter[0]
        window_width = ds.WindowWidth if isinstance(ds.WindowWidth, float) else ds.WindowWidth[0]
    else:
        window_center = np.mean(image_array)
        window_width = np.max(image_array) - np.min(image_array)
    
    # 画像データにウィンドウ幅とウィンドウ中心を適用
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    image_8bit = np.clip(image_array, min_val, max_val)
    image_8bit = ((image_8bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # 画像データをPIL Imageオブジェクトに変換
    image = Image.fromarray(image_8bit)
    
    # 出力ファイル名を生成
    output_file = os.path.join(output_dir, f'{file_name}.tiff')
    
    # 画像をTIFF形式で保存（16ビット）
    image.save(output_file, 'TIFF')

print(f'Converted {len(dicom_files)} DICOM files to TIFF format.')