#GUI/gui.py
import sys
import os
import re
import shutil
import pydicom
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, QTreeWidget, 
                             QTreeWidgetItem, QTextEdit, QProgressBar, QMessageBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QWheelEvent, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF
from ultralytics import YOLO
import pandas as pd

# 既存のコードの先頭に追加
AVAILABLE_SERIES = [
    "eT1W_SE_tra",
    "T2 TIME sue1 rfa180",
    "eT1W_SE_cor",
    "eT1W_SE_sag",
    "T2W_SPIR_cor"
]

model_path = 'best.pt'

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def get_label_colors():
    return {
        "l_so": (255, 0, 0),    # 赤
        "r_so": (255, 0, 0),    # 赤
        "l_io": (0, 255, 0),    # 緑
        "r_io": (0, 255, 0),    # 緑
        "l_sr": (0, 0, 255),    # 青
        "r_sr": (0, 0, 255),    # 青
        "l_ir": (255, 255, 0),  # 黄
        "r_ir": (255, 255, 0),  # 黄
        "l_lr": (255, 0, 255),  # マゼンタ
        "r_lr": (255, 0, 255),  # マゼンタ
        "l_mr": (0, 255, 255),  # シアン
        "r_mr": (0, 255, 255),  # シアン
        "so": (255, 0, 0),
        "io": (0, 255, 0),
        "sr": (0, 0, 255),   
        "ir": (255, 255, 0), 
        "lr": (255, 0, 255), 
        "mr": (0, 255, 255)
    }

def predict_and_visualize_from_dicom(dicom_file_path, model):
    dicom = pydicom.dcmread(dicom_file_path)
    img_width = dicom.Columns
    img_height = dicom.Rows
    pixel_spacing = dicom.PixelSpacing
    ds = pixel_spacing[1] * pixel_spacing[0]  # mm^2 微少面積
    pixel_array = dicom.pixel_array

    if pixel_array.dtype != np.uint8:
        pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

    if len(pixel_array.shape) == 2:
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)

    img = pixel_array.copy()
    results = model(img)
    label_colors = get_label_colors()
    label_areas = {}

    if results[0].masks is not None:
        for i, mask in enumerate(results[0].masks.cpu().xy):
            mask_array = np.array(mask, dtype=np.int32)
            mask_image = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.fillPoly(mask_image, [mask_array], 1)
            mask_area = cv2.countNonZero(mask_image) * ds

            M = cv2.moments(mask_image)
            center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else img_width // 2
            
            original_label = results[0].names[int(results[0].boxes.cls[i])]
            label = f"l_{original_label}" if center_x < img_width // 2 else f"r_{original_label}"

            color = label_colors.get(label, (128, 128, 128))  # デフォルトカラーをグレーに設定
            
            if label not in label_areas:
                label_areas[label] = 0
            label_areas[label] += mask_area
            
            pos = mask_array.reshape((-1,1,2))
            cv2.polylines(img, [pos], isClosed=True, color=color, thickness=2)
            x, y = pos.min(axis=0)[0]
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img, results, label_areas

class BatchProcessor(QThread):
    progress_updated = pyqtSignal(int)
    processing_complete = pyqtSignal(dict)

    def __init__(self, dicom_files, model):
        super().__init__()
        self.dicom_files = dicom_files
        self.model = model

    def run(self):
        all_results = {}
        total_files = len(self.dicom_files)
        for i, file_path in enumerate(self.dicom_files):
            _, _, label_areas = predict_and_visualize_from_dicom(file_path, self.model)
            all_results[os.path.basename(file_path)] = label_areas
            self.progress_updated.emit(int((i + 1) / total_files * 100))
        self.processing_complete.emit(all_results)

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
        else:
            factor = 0.8
        self.scale(factor, factor)

class WheelNavigableTreeWidget(QTreeWidget):
    itemChangedByWheel = pyqtSignal(object, int)  # カスタムシグナル

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.dicom_items = []  # DICOMファイルアイテムのリスト

    def wheelEvent(self, event):
        modifiers = event.modifiers()
        if modifiers == Qt.NoModifier:  # モディファイアキーが押されていない場合
            # 現在選択されているアイテムを取得
            current_item = self.currentItem()
            if current_item and self.dicom_items:
                # 現在のインデックスを取得
                current_index = self.dicom_items.index(current_item) if current_item in self.dicom_items else 0
                
                # 新しいインデックスを計算
                if event.angleDelta().y() > 0:
                    # 上にスクロール
                    new_index = max(0, current_index - 1)
                else:
                    # 下にスクロール
                    new_index = min(len(self.dicom_items) - 1, current_index + 1)
                
                # 新しいアイテムを選択
                new_item = self.dicom_items[new_index]
                self.setCurrentItem(new_item)
                self.scrollToItem(new_item)
                self.itemChangedByWheel.emit(new_item, 0)
                event.accept()
                return
        super().wheelEvent(event)

    def update_dicom_items(self):
        """ツリー内のすべてのDICOMファイルアイテムを収集"""
        self.dicom_items = []
        self._collect_dicom_items(self.invisibleRootItem())

    def _collect_dicom_items(self, parent_item):
        """再帰的にDICOMファイルアイテムを収集"""
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            # ファイルパスを持つアイテム（DICOMファイル）のみを追加
            if child.data(0, Qt.UserRole):
                self.dicom_items.append(child)
            if child.childCount() > 0:
                self._collect_dicom_items(child)

class DICOMProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dicom_files = []
        self.current_image = None
        self.model = self.load_yolo_model(model_path)
        self.all_results = {}
        self.label_colors = get_label_colors()
        self.need_fix_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'needFix')
        self.graphics_view = ZoomableGraphicsView(self)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setMinimumSize(800, 600)
        self.current_series = None
        self.selected_folder = None
        self.initUI()
    
    def load_yolo_model(self, model_path):
        # YOLOモデルを安全に読み込む
        model = YOLO(model_path)
        # model.model.cuda()  # GPUを使用する場合
        model.model.eval()  # 評価モードに設定
        return model

    def initUI(self):
        self.setWindowTitle('DICOMフォルダプロセッサ')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Series選択用のコンボボックスを追加
        series_layout = QHBoxLayout()
        series_label = QLabel('Series選択:', self)
        self.series_selector = QComboBox(self)
        self.series_selector.addItems(AVAILABLE_SERIES)
        self.series_selector.currentTextChanged.connect(self.on_series_changed)
        series_layout.addWidget(series_label)
        series_layout.addWidget(self.series_selector)
        left_layout.addLayout(series_layout)

        self.folder_label = QLabel('フォルダが選択されていません', self)
        left_layout.addWidget(self.folder_label)

        self.btn_select = QPushButton('DICOMフォルダを選択', self)
        self.btn_select.clicked.connect(self.select_folder)
        left_layout.addWidget(self.btn_select)

        # self.btn_export = QPushButton('全画像を処理してエクセルに出力', self)
        # self.btn_export.clicked.connect(self.process_and_export)
        # left_layout.addWidget(self.btn_export)

        # self.progress_bar = QProgressBar(self)
        # left_layout.addWidget(self.progress_bar)

        # 先にファイルツリーを作成
        self.file_tree = WheelNavigableTreeWidget(self)
        self.file_tree.setHeaderLabel('DICOM Files')
        self.file_tree.itemClicked.connect(self.display_selected_image)
        self.file_tree.itemChangedByWheel.connect(self.display_selected_image)

        # フォルダ操作ボタンのレイアウト（file_tree作成後に配置）
        folder_control_layout = QHBoxLayout()
        
        self.btn_expand_all = QPushButton('フォルダをすべて展開', self)
        self.btn_expand_all.clicked.connect(self.file_tree.expandAll)
        folder_control_layout.addWidget(self.btn_expand_all)
        
        self.btn_collapse_all = QPushButton('フォルダをすべて折りたたむ', self)
        self.btn_collapse_all.clicked.connect(self.file_tree.collapseAll)
        folder_control_layout.addWidget(self.btn_collapse_all)
        
        left_layout.addLayout(folder_control_layout)
        left_layout.addWidget(self.file_tree)

        self.graphics_view = ZoomableGraphicsView(self)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setMinimumSize(800, 600)
        right_layout.addWidget(self.graphics_view)

        self.image_info_label = QLabel(self)
        right_layout.addWidget(self.image_info_label)

        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        right_layout.addWidget(self.result_text)

        self.btn_need_fix = QPushButton('要修正フォルダに追加', self)
        self.btn_need_fix.clicked.connect(self.add_to_need_fix)
        right_layout.addWidget(self.btn_need_fix)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def get_series_info_from_data_txt(self, selected_folder, series):
        """data.txtからSeries情報を取得し、SEフォルダとのマッピングを作成"""
        # data.txtは選択フォルダの親フォルダにある
        parent_folder = os.path.dirname(selected_folder)
        data_txt_path = os.path.join(parent_folder, "data.txt")
        
        if not os.path.exists(data_txt_path):
            self.result_text.append(f"data.txtが見つかりません: {data_txt_path}")
            return {}

        try:
            with open(data_txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            series_info = {}
            # Series名のパターンを作成
            p = re.compile(r".*\[{}\].*Directory: (.*)".format(series))
            
            for line in lines:
                line = line.strip()
                match = p.match(line)
                if match:
                    # Directory: 以降のパスを取得
                    dir_path = match.group(1)
                    
                    # DATAフォルダ以降のパスを抽出
                    # DATA\147752\20240508\112048\EX9\SE5 から
                    # 147752\20240508\112048\EX9\SE5 を取得
                    if dir_path.startswith('DATA\\'):
                        relative_path = dir_path[5:]  # "DATA\"の部分を除去
                        
                        # 選択されたフォルダと組み合わせて完全なパスを作成
                        full_path = os.path.join(selected_folder, relative_path)
                        
                        # EXとSEフォルダ名を抽出
                        folder_parts = relative_path.split('\\')
                        ex_se_parts = [part for part in folder_parts if part.startswith(('EX', 'SE'))]
                        if len(ex_se_parts) == 2:
                            ex_folder, se_folder = ex_se_parts
                            
                            # キーとなるEX/SEの組み合わせを作成
                            key = f"{ex_folder}_{se_folder}"
                            series_info[key] = {
                                'se_folder': se_folder,
                                'ex_folder': ex_folder,
                                'full_path': full_path,
                                'dicom_files': []
                            }
                            
                            # DICOMファイルを確認
                            if os.path.exists(full_path):
                                dicom_files = [f for f in os.listdir(full_path) if re.match(r'IMG\d+$', f)]
                                series_info[key]['dicom_files'] = sorted(dicom_files, key=natural_sort_key)
                            else:
                                self.result_text.append(f"警告: フォルダが見つかりません: {full_path}")

            # 有効なDICOMファイルを持つエントリのみを返す
            valid_info = {k: v for k, v in series_info.items() if v['dicom_files']}
            self.result_text.append(f"Found {len(valid_info)} valid folders")
            return valid_info

        except Exception as e:
            self.result_text.append(f"エラーが発生しました: {str(e)}")
            return {}

    def update_file_tree_with_series(self):
        """選択されたSeriesに基づいてファイルツリーを更新"""
        self.file_tree.clear()
        self.dicom_files = []
        
        if not self.current_series or not self.selected_folder:
            return

        # Series情報を取得
        series_info = self.get_series_info_from_data_txt(self.selected_folder, self.current_series)
        
        if not series_info:
            self.result_text.append(f"選択されたSeries ({self.current_series}) に対応するフォルダが見つかりませんでした。")
            return

        # ツリーを構築
        for key, info in series_info.items():
            # SEフォルダ名とEXフォルダ名を使用して表示名を作成
            display_name = f"{info['se_folder']} ({info['ex_folder']})"
            
            folder_item = QTreeWidgetItem([display_name])
            self.file_tree.addTopLevelItem(folder_item)
            
            # DICOMファイルを追加
            for file_name in info['dicom_files']:
                file_path = os.path.join(info['full_path'], file_name)
                self.dicom_files.append(file_path)
                
                file_item = QTreeWidgetItem([file_name])
                file_item.setData(0, Qt.UserRole, file_path)
                folder_item.addChild(file_item)

        # DICOMファイルアイテムのリストを更新
        self.file_tree.update_dicom_items()
        
        # 最初のDICOMファイルを自動選択
        if self.dicom_files:
            self.select_first_dicom_file()
            self.expand_to_selected_item()
            self.result_text.append(f"{len(self.dicom_files)} 個のDICOMファイルが見つかりました。")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'DATAフォルダを選択')
        if folder:
            # 選択されたフォルダの名前がDATAであることを確認
            if os.path.basename(folder) != 'DATA':
                QMessageBox.warning(self, '警告', 'DATAフォルダを選択してください。')
                return
                
            self.selected_folder = folder
            self.folder_label.setText(f'選択されたフォルダ: {folder}')
            if self.current_series:
                self.update_file_tree_with_series()

    def on_series_changed(self, series):
        """Seriesが変更されたときの処理"""
        self.current_series = series
        if self.selected_folder:
            self.update_file_tree_with_series()

    def expand_to_selected_item(self):
        """選択されているアイテムまでのパスを展開する"""
        current_item = self.file_tree.currentItem()
        if current_item:
            # 親アイテムを取得して展開
            parent = current_item.parent()
            while parent:
                parent.setExpanded(True)
                parent = parent.parent()
            # 選択したアイテムが見えるようにスクロール
            self.file_tree.scrollToItem(current_item)
    
    def select_first_dicom_file(self):
        """最初のDICOMファイルを見つけて選択する"""
        if self.file_tree.dicom_items:
            first_item = self.file_tree.dicom_items[0]
            self.file_tree.setCurrentItem(first_item)
            self.display_selected_image(first_item, 0)

    def process_and_export(self):
        if not self.dicom_files:
            self.result_text.append("処理するファイルがありません。フォルダを選択してください。")
            return

        self.progress_bar.setValue(0)
        self.btn_export.setEnabled(False)
        self.result_text.append("処理を開始します...")

        self.batch_processor = BatchProcessor(self.dicom_files, self.model)
        self.batch_processor.progress_updated.connect(self.update_progress)
        self.batch_processor.processing_complete.connect(self.on_processing_complete)
        self.batch_processor.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_processing_complete(self, results):
        self.all_results = results
        self.result_text.append("全画像の処理が完了しました。エクセルファイルの保存先を選択してください。")
        self.export_to_excel()

    def export_to_excel(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "エクセルファイルを保存", "", "Excel Files (*.xlsx)")
        if save_path:
            try:
                df = pd.DataFrame(self.all_results).T
                df.index.name = 'ファイル名'
                
                columns = sorted(df.columns, key=lambda x: (x[2:], x[0]))
                df = df.reindex(columns=columns)
                
                df.to_excel(save_path)
                self.result_text.append(f"結果を {save_path} に出力しました。")
            except Exception as e:
                self.result_text.append(f"エラーが発生しました: {str(e)}")
        else:
            self.result_text.append("エクセルファイルの保存がキャンセルされました。")

        self.btn_export.setEnabled(True)

    def display_selected_image(self, item, column):
        # ファイルパスを取得
        file_path = item.data(0, Qt.UserRole)
        if file_path:
            self.process_and_display_image(file_path)

    def process_and_display_image(self, file_path):
        self.current_image = file_path
        processed_image, results, label_areas = predict_and_visualize_from_dicom(self.current_image, self.model)

        height, width, channel = processed_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.graphics_scene.clear()
        pixmap_item = self.graphics_scene.addPixmap(pixmap)
        self.graphics_scene.setSceneRect(QRectF(pixmap.rect()))
        self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)

        # 画像情報と結果の表示を更新
        dicom = pydicom.dcmread(self.current_image)
        self.image_info_label.setText(f"画像サイズ: {width}x{height}\n"
                                      f"患者ID: {dicom.PatientID if 'PatientID' in dicom else '不明'}\n"
                                      f"検査日: {dicom.StudyDate if 'StudyDate' in dicom else '不明'}")

        self.result_text.clear()
        self.result_text.append("検出結果:")
        for label, area in label_areas.items():
            self.result_text.append(f"- {label}: 面積 {area:.2f} mm2")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'graphics_view') and hasattr(self, 'graphics_scene'):
            self.graphics_view.fitInView(self.graphics_scene.sceneRect(), Qt.KeepAspectRatio)

    def add_to_need_fix(self):
        if self.current_image:
            try:
                file_path = self.current_image
                file_name = os.path.basename(file_path)  # IMG****
                
                # パスからEXとSE番号を取得
                path_parts = os.path.normpath(file_path).split(os.sep)
                ex_folder = next((part for part in path_parts if part.startswith('EX')), '')
                se_folder = next((part for part in path_parts if part.startswith('SE')), '')
                
                if not ex_folder or not se_folder:
                    QMessageBox.warning(self, "エラー", "EXフォルダまたはSEフォルダが見つかりません。")
                    return
                
                # 新しいファイル名を作成（例：EX1SE1IMG1）
                img_number = re.search(r'IMG(\d+)', file_name)
                if img_number:
                    new_file_name = f"{ex_folder}{se_folder}IMG{img_number.group(1)}"
                else:
                    QMessageBox.warning(self, "エラー", "画像番号の取得に失敗しました。")
                    return
                
                destination = os.path.join(self.need_fix_folder, new_file_name)
                
                # 既に同じ名前のファイルが存在する場合の処理
                if os.path.exists(destination):
                    overwrite = QMessageBox.question(self, 
                                                   "確認", 
                                                   f"ファイル {new_file_name} は既に存在します。上書きしますか？",
                                                   QMessageBox.Yes | QMessageBox.No)
                    if overwrite == QMessageBox.No:
                        return
                
                shutil.copy2(file_path, destination)
                
                message = (f"ファイルを要修正フォルダに追加しました。\n"
                          f"新しいファイル名: {new_file_name}\n"
                          f"保存先: {destination}")
                QMessageBox.information(self, "成功", message)
                
            except Exception as e:
                error_message = f"ファイルの追加中にエラーが発生しました: {str(e)}"
                QMessageBox.warning(self, "エラー", error_message)
        else:
            warning_message = "画像が選択されていません。"
            QMessageBox.warning(self, "警告", warning_message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DICOMProcessor()
    ex.show()
    sys.exit(app.exec_())