import cv2
from ultralytics import YOLO
import numpy as np
import pydicom

#変更するところ
dir = 'runs/segment/train/' 
model = YOLO(dir+'weights/best.pt') 
dcm = "DATA/EXFILES/EX15/SE3/IMG7"


def predict_and_visualize_from_dicom(dicom_file_path, model,area=False):
  # DICOMファイルを読み込む
  dicom = pydicom.dcmread(dicom_file_path)
  img_width = dicom.Columns
  img_height = dicom.Rows
  pixel_spacing = dicom.PixelSpacing
  ds= pixel_spacing[1]*pixel_spacing[0] #mm^2 微少面積
  # ピクセルデータを取得
  pixel_array = dicom.pixel_array

  # 画像の正規化 (0-255の範囲に変換)
  if pixel_array.dtype != np.uint8:
    pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

  # グレースケールの場合、3チャンネルに変換
  if len(pixel_array.shape) == 2:
    pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)

  # 元の画像のコピーを作成（描画用）
  img = pixel_array.copy()

  # 予測を実行
  results = model(img, save=True, save_txt=True, agnostic_nms=True)

  # マスクの描画
  if results[0].masks is not None:
    for i, mask in enumerate(results[0].masks.cpu().xy):
      # マスクの座標を整数に変換
      mask_array = np.array(mask, dtype=np.int32)
      if area:
        # 面積を計算
        mask_image = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask_image, [mask_array], 1)
        mask_area = cv2.countNonZero(mask_image) *ds
        label = labels[int(classes[i])]  
        print(f"Mask {i} ({label}) area: {mask_area} mm²")
      # 描画
      pos = mask_array.reshape((-1,1,2))
      color = (255, 0, 100 * (i % 3))  # 色のバリエーションを増やす
      cv2.polylines(img, [pos], isClosed=True, color=color, thickness=2)
      if area:
        # ラベル名の表示
        x, y = pos.min(axis=0)[0] # マスクの左上の座標を取得
        label_text = f"{label}: {format(mask_area, '.2f')} mm2" # ラベルのテキスト
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)# テキストの背景色（黒い半透明の背景）
        cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y), color, -1)
        cv2.putText(img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)# テキストを白色で描画
  return img, results


print("------------results------------")
img,results = predict_and_visualize_from_dicom(dcm, model)
# ラベル情報を取得
labels = results[0].names
classes = results[0].boxes.cls
print("results:"+results)
print(labels)
print(classes)

# 結果の表示
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(dir+'result_without_labels.jpg', img)

img,results = predict_and_visualize_from_dicom(dcm, model,True)
# 結果の表示
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(dir+'result_with_labels.jpg', img)