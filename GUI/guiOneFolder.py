#GUI/guiOneFolder.py
import sys
import os
import re
import shutil
import pydicom
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, QTreeWidget, QTreeWidgetItem, QTextEdit, QProgressBar, QMessageBox,QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PyQt5.QtGui import QImage, QPixmap, QWheelEvent,QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF
from ultralytics import YOLO
import pandas as pd

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
        # image_labelの代わりにgraphics_viewを使用
        self.graphics_view.setMinimumSize(800, 600)
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

        self.folder_label = QLabel('フォルダが選択されていません', self)
        left_layout.addWidget(self.folder_label)

        self.btn_select = QPushButton('DICOMフォルダを選択', self)
        self.btn_select.clicked.connect(self.select_folder)
        left_layout.addWidget(self.btn_select)

        self.btn_export = QPushButton('全画像を処理してエクセルに出力', self)
        self.btn_export.clicked.connect(self.process_and_export)
        left_layout.addWidget(self.btn_export)

        self.progress_bar = QProgressBar(self)
        left_layout.addWidget(self.progress_bar)

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

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'DICOMフォルダを選択')
        if folder:
            self.folder_label.setText(f'選択されたフォルダ: {folder}')
            self.dicom_files = []
            self.file_tree.clear()
            self.all_results = {}

            # フォルダ構造を走査してDICOMファイルを探す
            for root, dirs, files in os.walk(folder):
                # パスの相対部分を取得
                rel_path = os.path.relpath(root, folder)
                if rel_path == '.':
                    current_parent = None
                else:
                    # フォルダパスを'/'で分割
                    path_parts = rel_path.split(os.sep)
                    current_parent = None
                    
                    # ツリー構造を構築
                    for part in path_parts:
                        # 現在の階層でフォルダを探す
                        found_item = None
                        if current_parent is None:
                            # トップレベルの項目を探す
                            for i in range(self.file_tree.topLevelItemCount()):
                                item = self.file_tree.topLevelItem(i)
                                if item.text(0) == part:
                                    found_item = item
                                    break
                        else:
                            # 子アイテムを探す
                            for i in range(current_parent.childCount()):
                                item = current_parent.child(i)
                                if item.text(0) == part:
                                    found_item = item
                                    break

                        if found_item is None:
                            # 新しいフォルダ項目を作成
                            new_item = QTreeWidgetItem([part])
                            if current_parent is None:
                                self.file_tree.addTopLevelItem(new_item)
                            else:
                                current_parent.addChild(new_item)
                            current_parent = new_item
                        else:
                            current_parent = found_item

                # DICOMファイルをフィルタリングして追加
                dicom_files = [f for f in files if re.match(r'IMG\d+$', f)]
                if dicom_files:
                    # ファイルを自然順でソート
                    dicom_files.sort(key=natural_sort_key)
                    for file in dicom_files:
                        full_path = os.path.join(root, file)
                        self.dicom_files.append(full_path)
                        file_item = QTreeWidgetItem([file])
                        file_item.setData(0, Qt.UserRole, full_path)
                        if current_parent is None:
                            self.file_tree.addTopLevelItem(file_item)
                        else:
                            current_parent.addChild(file_item)
            
            # ツリーは最初は折りたたんだ状態にする
            self.file_tree.collapseAll()
            
            # DICOMファイルアイテムのリストを更新
            self.file_tree.update_dicom_items()
            
            # needFixフォルダを作成
            if not os.path.exists(self.need_fix_folder):
                os.makedirs(self.need_fix_folder)

            # 最初のDICOMファイルを自動選択
            if self.dicom_files:
                self.select_first_dicom_file()
                # 選択したアイテムの親フォルダを展開
                self.expand_to_selected_item()
        
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
                # 現在の画像ファイルのパスから必要な情報を取得
                file_path = self.current_image
                # 元のフォルダ構造を取得（2階層分）
                base_path = os.path.dirname(os.path.dirname(file_path))  # 2階層上
                rel_path = os.path.relpath(file_path, base_path)  # 相対パス取得
                path_parts = rel_path.split(os.sep)  # パスを分割
                
                # 新しいファイル名を作成（フォルダ名_ファイル名）
                if len(path_parts) >= 2:
                    new_file_name = f"{path_parts[0]}_{path_parts[1]}"
                else:
                    new_file_name = os.path.basename(file_path)
                
                destination = os.path.join(self.need_fix_folder, new_file_name)
                shutil.copy2(file_path, destination)
                
                message = f"ファイルを要修正フォルダに追加しました。\n新しいファイル名: {new_file_name}\n保存先: {destination}"
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