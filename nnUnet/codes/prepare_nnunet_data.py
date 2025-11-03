"""
nnU-Net用データ準備スクリプト（改良版）
YOLOで使用していたLabelMe形式のJSONデータをnnU-Net形式に変換
"""
import os
import glob
import json
import shutil
import numpy as np
from PIL import Image
import pydicom
import SimpleITK as sitk
from typing import Dict, List, Tuple
import argparse

# === 設定項目 ===
TASK_ID = 501  # 眼筋セグメンテーション用タスクID
TASK_NAME = "EyeMuscleSegmentation"
TASK_FOLDER_NAME = f"Dataset{TASK_ID:03d}_{TASK_NAME}"

# ラベルの定義（6つの眼筋）
LABEL_MAP = {
    "so": 1,  # 上斜筋
    "io": 2,  # 下斜筋  
    "sr": 3,  # 上直筋
    "ir": 4,  # 下直筋
    "lr": 5,  # 外直筋
    "mr": 6,  # 内直筋
}

# 逆引きマップ（0=background を含む）
LABEL_NAMES = ["background"] + list(LABEL_MAP.keys())


def setup_directories(base_dir: str) -> Tuple[str, str, str, str]:
    """nnU-Net用のディレクトリ構造を作成"""
    output_dir = os.path.join(base_dir, TASK_FOLDER_NAME)
    imagesTr_dir = os.path.join(output_dir, "imagesTr")
    labelsTr_dir = os.path.join(output_dir, "labelsTr")
    imagesTs_dir = os.path.join(output_dir, "imagesTs")
    
    # 既存のディレクトリをクリーンアップ
    if os.path.exists(output_dir):
        print(f"既存のディレクトリを削除: {output_dir}")
        shutil.rmtree(output_dir)
    
    # 新規作成
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)
    os.makedirs(imagesTs_dir, exist_ok=True)
    
    print(f"ディレクトリを作成しました: {output_dir}")
    return output_dir, imagesTr_dir, labelsTr_dir, imagesTs_dir


def json_to_mask(json_path: str, label_map: Dict) -> Tuple[np.ndarray, Dict]:
    """LabelMe JSONからマスク画像を生成"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    height = data['imageHeight']
    width = data['imageWidth']
    
    # 空のマスクを作成
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 各シェイプを処理
    for shape in data['shapes']:
        label = shape['label']
        
        # l_/r_プレフィックスを除去
        if label.startswith('l_') or label.startswith('r_'):
            label = label[2:]
        
        if label in label_map:
            points = np.array(shape['points'], dtype=np.int32)
            
            # OpenCVを使ってポリゴンを塗りつぶし
            import cv2
            cv2.fillPoly(mask, [points], label_map[label])
    
    return mask, data


def process_dicom_to_nifti(dicom_path: str) -> sitk.Image:
    """DICOMファイルをSimpleITK画像に変換"""
    ds = pydicom.dcmread(dicom_path)
    image_array = ds.pixel_array
    
    # ウィンドウ処理（表示用の正規化）
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        window_center = ds.WindowCenter if isinstance(ds.WindowCenter, (int, float)) else ds.WindowCenter[0]
        window_width = ds.WindowWidth if isinstance(ds.WindowWidth, (int, float)) else ds.WindowWidth[0]
    else:
        window_center = np.mean(image_array)
        window_width = np.max(image_array) - np.min(image_array)
    
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    image_array = np.clip(image_array, min_val, max_val)
    
    # 正規化（0-1の範囲に）
    image_array = (image_array - min_val) / (max_val - min_val)
    
    # SimpleITKイメージに変換（2D → 3D）
    image_array = image_array.astype(np.float32)
    image_array = np.expand_dims(image_array, axis=-1)  # (H, W) -> (H, W, 1)
    sitk_image = sitk.GetImageFromArray(image_array.transpose(2, 1, 0))  # (Z, Y, X)
    
    # スペーシング情報を設定
    if hasattr(ds, 'PixelSpacing'):
        spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), 1.0]
        sitk_image.SetSpacing(spacing)
    
    return sitk_image


def convert_tiff_to_nifti(tiff_path: str) -> sitk.Image:
    """TIFFファイルをSimpleITK画像に変換"""
    img = Image.open(tiff_path)
    img_array = np.array(img).astype(np.float32) / 255.0  # 正規化
    
    # 2D → 3D
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W)
    sitk_image = sitk.GetImageFromArray(img_array)
    
    return sitk_image


def process_dataset(json_dir: str, image_dir: str, output_dirs: Tuple, 
                   train_ratio: float = 0.8) -> Dict:
    """データセット全体を処理"""
    _, imagesTr_dir, labelsTr_dir, imagesTs_dir = output_dirs
    
    # JSONファイルを取得
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    print(f"見つかったJSONファイル: {len(json_files)}個")
    
    # エラー回避用 改善の余地あり
    print("480x480 でないTIFFファイルを除外するためにフィルタリングします...")
    filtered_json_files = []
    skipped_count = 0
    
    for json_path in json_files:
        base_name = os.path.basename(json_path).replace('.json', '')
        tiff_path = os.path.join(image_dir, f"{base_name}.tiff")
        
        # まず、TIFFファイルが存在するかどうかをチェック
        if os.path.exists(tiff_path):
            try:
                # SimpleITKで画像の情報を読み込む (画像全体を読み込むより高速)
                reader = sitk.ImageFileReader()
                reader.SetFileName(tiff_path)
                reader.LoadPrivateTagsOff()
                reader.ReadImageInformation()
                
                size = reader.GetSize() # (Width, Height) または (Width, Height, Depth)
                
                # 2D画像 (サイズが (480, 480)) か
                # 3D画像 (サイズが (480, 480, 1)) かをチェック
                is_correct_size = False
                if len(size) == 2 and (size[0], size[1]) == (480, 480):
                    is_correct_size = True
                elif len(size) == 3 and (size[0], size[1]) == (480, 480) and size[2] == 1:
                    is_correct_size = True

                if not is_correct_size:
                    print(f"  スキップ: {base_name}.tiff は 480x480 ではありません (サイズ: {size})")
                    skipped_count += 1
                    continue # このJSONファイルはリストに追加しない
                    
            except Exception as e:
                print(f"  警告: {base_name}.tiff の読み込み中にエラー: {e}。スキップします。")
                skipped_count += 1
                continue
        
        # 480x480であるか、TIFFでない (DICOMなど) 場合は処理対象リストに追加
        filtered_json_files.append(json_path)
        
    print(f"フィルタリング完了。{skipped_count}個のファイルをスキップしました。")
    print(f"処理対象のJSONファイル: {len(filtered_json_files)}個")
    
    # 元のリストを、フィルタリング済みのリストに置き換える
    json_files = filtered_json_files

    # トレーニング/テスト分割
    num_train = int(len(json_files) * train_ratio)
    train_files = json_files[:num_train]
    test_files = json_files[num_train:]
    
    print(f"トレーニング: {len(train_files)}個, テスト: {len(test_files)}個")
    
    # 処理統計
    stats = {'train': [], 'test': []}
    
    # トレーニングデータ処理
    for idx, json_path in enumerate(train_files):
        case_name = f"case_{idx:04d}"
        print(f"処理中 (Train): {case_name} <- {os.path.basename(json_path)}")
        
        # マスク生成
        mask, json_data = json_to_mask(json_path, LABEL_MAP)
        
        # 対応する画像を探す
        base_name = os.path.basename(json_path).replace('.json', '')
        
        # まずTIFFを探す
        tiff_path = os.path.join(image_dir, f"{base_name}.tiff")
        if os.path.exists(tiff_path):
            sitk_image = convert_tiff_to_nifti(tiff_path)
        else:
            # DICOMを探す（needFixフォルダなど）
            dcm_patterns = [
                os.path.join(image_dir, f"{base_name}"),
                os.path.join(image_dir, f"{base_name}.dcm"),
            ]
            sitk_image = None
            for dcm_path in dcm_patterns:
                if os.path.exists(dcm_path):
                    sitk_image = process_dicom_to_nifti(dcm_path)
                    break
            
            if sitk_image is None:
                print(f"  警告: 画像ファイルが見つかりません: {base_name}")
                continue
        
        # マスクをSimpleITK形式に変換
        mask_3d = np.expand_dims(mask, axis=0)  # (1, H, W)
        sitk_mask = sitk.GetImageFromArray(mask_3d)
        sitk_mask.CopyInformation(sitk_image)  # スペーシング等をコピー
        
        # 保存
        sitk.WriteImage(sitk_image, os.path.join(imagesTr_dir, f"{case_name}_0000.nii.gz"))
        sitk.WriteImage(sitk_mask, os.path.join(labelsTr_dir, f"{case_name}.nii.gz"))
        
        stats['train'].append(case_name)
    
    # テストデータ処理（ラベルなし）
    for idx, json_path in enumerate(test_files):
        case_name = f"case_{idx:04d}"
        print(f"処理中 (Test): {case_name} <- {os.path.basename(json_path)}")
        
        # 画像のみ処理（テストデータはラベル不要）
        base_name = os.path.basename(json_path).replace('.json', '')
        tiff_path = os.path.join(image_dir, f"{base_name}.tiff")
        
        if os.path.exists(tiff_path):
            sitk_image = convert_tiff_to_nifti(tiff_path)
            sitk.WriteImage(sitk_image, os.path.join(imagesTs_dir, f"{case_name}_0000.nii.gz"))
            stats['test'].append(case_name)
    
    return stats


def create_dataset_json(output_dir: str, num_training: int) -> None:
    """nnU-Net用のdataset.jsonを生成"""
    dataset_json = {
        "channel_names": {
            "0": "MRI"  # _0000.nii.gz に対応
        },
        "labels": {str(i): name for i, name in enumerate(LABEL_NAMES)},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "regions_class_order": list(range(1, len(LABEL_NAMES))),  # 背景を除く
    }
    
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"dataset.jsonを生成しました: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLOデータをnnU-Net形式に変換')
    default_data_dir = "C:/Users/mitae/workspace/imageProcessing/datasets/2024_07_11_09_34_02/filtered_json4"
    parser.add_argument('--json_dir', type=str, default=default_data_dir,
                       help='JSONアノテーションファイルのディレクトリ')
    parser.add_argument('--image_dir', type=str, default=default_data_dir,
                       help='画像ファイル（TIFF/DICOM）のディレクトリ')
    parser.add_argument('--output_base', type=str, default="C:/Users/mitae/workspace/imageProcessing/nnUnet/raw_data",
                       help='nnU-Netのraw dataディレクトリ')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='トレーニングデータの割合')
    
    args = parser.parse_args()
    
    # 環境変数の確認
    if not os.environ.get('nnUNet_raw'):
        print("警告: 環境変数 nnUNet_raw が設定されていません")
        print(f"デフォルトパスを使用: {args.output_base}")
        os.environ['nnUNet_raw'] = args.output_base
    
    # ディレクトリセットアップ
    output_dirs = setup_directories(args.output_base)
    
    # データ変換
    stats = process_dataset(args.json_dir, args.image_dir, output_dirs, args.train_ratio)
    
    # dataset.json生成
    create_dataset_json(output_dirs[0], len(stats['train']))
    
    print("\n=== 変換完了 ===")
    print(f"トレーニングケース: {len(stats['train'])}個")
    print(f"テストケース: {len(stats['test'])}個")
    print(f"出力先: {output_dirs[0]}")
    print("\n次のステップ:")
    print("1. nnUNet_plan_and_preprocess -t 501")
    print("2. nnUNet_train 2d nnUNetTrainerV2 Task501_EyeMuscleSegmentation 0")


if __name__ == "__main__":
    main()
