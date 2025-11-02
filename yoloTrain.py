# ->Predict Phase
# たくさん画像いれてgui.pyで良くないやつをピックアップしてneedFixへ
# convertToTiff.pyでneedFixからTiff画像にしてTiffsに保存。
# Tiffにしたらノートパソコンから取り出してCVATにあげる ->Annotation Phase。
# CVATでアノテーションしまくる
# Annotation Phase -> Training Phase
# xml達を DATA/xmls に入れる
# labelmexml2json.py xml形式からjsonに
# segment_filter.py 必要なラベルだけにする
# yoloTrain.pyでTraining ※segment_filter.pyでyamlファイルが変わってるからここで変えることはない
# -> Predict Phase
# predict.py で試してみて良さげだったらbest.pyを取り出してノートパソコンへ入れる。

import os
import torch
from ultralytics import YOLO
import multiprocessing

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # CUDA設定の詳細を表示
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

    # Load a model
    model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

    # デバイスの明示的な指定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Train the model
        results = model.train(
            data="dataset.yaml", 
            epochs=500, 
            imgsz=480,
            device=device,
            workers=0,  # Windowsでのマルチプロセッシング問題を回避
            verbose=True  # 詳細な学習ログを表示
        )

        # トレーニング結果の確認
        print("Model saved to:", results.save_dir)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()