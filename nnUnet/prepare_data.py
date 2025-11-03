## nnUnet/prepare_data.py
import os
import glob
import json
import shutil
import numpy as np
from PIL import Image
import labelme
import SimpleITK as sitk

# --- 1. 設定項目 ---

# タスクID (001から999の間で、他のタスクと被らないように設定)
TASK_ID = 101
TASK_NAME = "MyEyeSegmentation"
TASK_FOLDER_NAME = f"Dataset{TASK_ID:03d}_{TASK_NAME}"

# あなたの元のデータがある場所
ORIGINAL_IMAGE_DIR = "C:/Users/mitae/workspace/imageProcessing/datasets/2024_07_11_09_34_02/filtered_json4"  # EX7SE3IMG07.tiff がある場所
JSON_ANNOTATION_DIR = "C:/Users/mitae/workspace/imageProcessing/datasets/2024_07_11_09_34_02/filtered_json4"   # .json がある場所

# nnU-Netがデータを読み込む場所 (環境変数 nnUNet_raw_data_base で設定したパス)
# このスクリプトは、このディレクトリ配下に Task101_MyEyeSegmentation フォルダを自動生成します
NNUNET_RAW_DATA_DIR = os.environ.get("nnUNet_raw")
NNUNET_RAW_DATA_DIR = os.path.normpath(NNUNET_RAW_DATA_DIR)
if NNUNET_RAW_DATA_DIR is None:
    print("エラー: 環境変数 'nnUNet_raw' が設定されていません。")
    print("例: export nnUNet_raw=\"/path/to/my_nnunet_data/nnUNet_raw_data\"")
    exit(1)

# 出力先ディレクトリのパス
output_dir = os.path.join(NNUNET_RAW_DATA_DIR, TASK_FOLDER_NAME)
imagesTr_dir = os.path.join(output_dir, "imagesTr")
labelsTr_dir = os.path.join(output_dir, "labelsTr")

# フォルダをクリーンアップ＆作成
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

print(f"nnU-Netタスクフォルダを作成しました: {output_dir}")

# --- 2. ラベルの定義 ---
# .json内の "label" と、マスク画像のピクセル値を対応させます
# 0 は "background" (背景) のために予約されています
label_map = {
"ir":1,
"mr":2,
"sr":3,
"so":4,
"lr":5,
"io":6,
}
# LabelMeに渡すための、逆引きマップ (0=background を含む)
label_names = ["background"] + list(label_map.keys())


# --- 3. データ変換ループ ---
training_cases = []
json_files = glob.glob(os.path.join(JSON_ANNOTATION_DIR, "*.json"))

for json_file_path in json_files:
    case_name = os.path.basename(json_file_path).replace(".json", "")
    
    # 1. JSONアノテーションを読み込む
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # imagePathから元の画像ファイル名を取得
        original_image_name = data["imagePath"]
        original_image_path = os.path.join(ORIGINAL_IMAGE_DIR, original_image_name)
        
        if not os.path.exists(original_image_path):
            print(f"警告: {original_image_path} が見つかりません。スキップします。")
            continue
            
        print(f"処理中: {case_name}")

        # 2. LabelMe JSONからマスク画像(numpy配列)を生成
        
        # JSONから画像サイズとシェイプを取得 (data は json.load(f) で読み込み済み)
        height = data['imageHeight']
        width = data['imageWidth']
        shapes = data['shapes']

        # ラベル名とIDのマップを作成 (background=0 を含む)
        label_name_to_value = {name: i for i, name in enumerate(label_names)}

        # labelme 4.x 系で標準的な shapes_to_label を使用
        try:
            # この関数は (mask_data, label_names_dict) のタプルを返します
            mask_data_tuple = labelme.utils.shapes_to_label(
                img_shape=(height, width),
                shapes=shapes,
                label_name_to_value=label_name_to_value
            )
            # タプルの1番目 (インデックス 0) がNumpy配列のマスクデータです
            mask_data = mask_data_tuple[0].astype(np.uint8)
        except Exception as e:
            print(f"  [エラー] マスクの生成に失敗しました: {e}")
            continue # このファイルの処理をスキップ
        
        # 3. 元画像を読み込む (nnU-NetはNIfTI形式を推奨)
        # 2D画像(tiff, png)の場合、SimpleITKを使ってNIfTIに変換します
        img_pil = Image.open(original_image_path)
        img_data = np.array(img_pil)
        
        # もし画像がRGBなら、チャネルを分離するかグレースケールにする必要があります
        # ここではグレースケール (L) と仮定します。
        # もしRGB (shapeが H, W, 3) の場合は、nnU-Netのチャネル設定(_0000, _0001, _0002) が必要です
        # 今回は2D画像、1チャネルと仮定して進めます
        if img_data.ndim == 3:
             print(f"警告: {original_image_name} はRGBです。グレースケールに変換します。")
             img_pil = img_pil.convert('L')
             img_data = np.array(img_pil)

        # 4. nnU-Net形式 (NIfTI) で保存
        # SimpleITKは (x, y, z) の順を好むため、必要に応じて転置
        # 2D画像なので (W, H) -> (W, H, 1) のようにZ軸を追加
        img_data = np.expand_dims(img_data, axis=-1) # (H, W, 1)
        img_data = img_data.transpose(1, 0, 2)       # (W, H, 1)
        sitk_img = sitk.GetImageFromArray(img_data)
        
        mask_data = np.expand_dims(mask_data, axis=-1) # (H, W, 1)
        mask_data = mask_data.transpose(1, 0, 2)       # (W, H, 1)
        sitk_mask = sitk.GetImageFromArray(mask_data)
        
        # ファイル名: {case_name}_0000.nii.gz (画像) と {case_name}.nii.gz (ラベル)
        # nnU-Net V2 では _0000 はチャネル（Modality）を示します
        sitk.WriteImage(sitk_img, os.path.join(imagesTr_dir, f"{case_name}_0000.nii.gz"))
        sitk.WriteImage(sitk_mask, os.path.join(labelsTr_dir, f"{case_name}.nii.gz"))
        
        training_cases.append(case_name)

    except Exception as e:
        print(f"エラー: {json_file_path} の処理に失敗しました。 {e}")

# --- 4. dataset.json を自動生成 ---
# nnU-Net V2 の学習に必須のファイルです
labels_dict = {str(val): key for key, val in label_map.items()}
labels_dict["0"] = "background" # 必須

dataset_json = {
    "channel_names": {
        "0": "image"  # _0000.nii.gz に対応
    },
    "labels": labels_dict,
    "numTraining": len(training_cases),
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO" # 2D NIfTIを正しく読み込むため
}

with open(os.path.join(output_dir, "dataset.json"), 'w') as f:
    json.dump(dataset_json, f, indent=4)

print("\n--- 完了 ---")
print(f"{len(training_cases)} 件のデータを変換し、{output_dir} に保存しました。")
print(f"dataset.json を自動生成しました。")
print("\n次のステップに進んでください。")