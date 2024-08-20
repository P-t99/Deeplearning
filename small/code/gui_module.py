import sys
import csv
import numpy as np
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QListWidget, QLineEdit, 
                             QMessageBox, QFileDialog, QSlider, QSplitter,
                             QDialog, QVBoxLayout)
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlaybackWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('数据回放')
        self.layout = QVBoxLayout()
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def update_heatmap(self, matrix):
        self.ax.clear()
        self.ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect=1.2)
        self.ax.set_title("回放热力图")
        self.ax.set_xlabel("X轴")
        self.ax.set_ylabel("Y轴")
        self.canvas.draw()

class DataCollectionGUI(QWidget):
    start_collection = pyqtSignal()
    pause_collection = pyqtSignal()
    stop_collection = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.initUI()
        self.collecting = False
        self.paused = False
        self.collected_data = []
        self.playback_data = []
        self.playback_index = 0
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_playback)
        self.heatmap_colorbar = None
        self.playback_window = None

    def initUI(self):
        self.setWindowTitle('睡姿数据采集')
        main_layout = QHBoxLayout()
        
        # Left panel for controls
        left_panel = QVBoxLayout()
        
        # Web info display
        self.web_info_label = QLabel('当前睡姿: 未知')
        self.confidence_label = QLabel('置信度: 0.00')
        self.timestamp_label = QLabel('最后更新时间: -')
        left_panel.addWidget(self.web_info_label)
        left_panel.addWidget(self.confidence_label)
        left_panel.addWidget(self.timestamp_label)

        # New information display for matrix metrics
        self.top48_avg_label = QLabel('Top48均值: -')
        self.rest_avg_label = QLabel('其余均值: -')
        self.difference_percentage_label = QLabel('差值百分比: -')
        left_panel.addWidget(self.top48_avg_label)
        left_panel.addWidget(self.rest_avg_label)
        left_panel.addWidget(self.difference_percentage_label)
        
        # 添加新的中位数标签
        self.top48_median_label = QLabel('Top48中位数: -')
        self.rest_median_label = QLabel('其余中位数: -')
        self.difference_percentage_median_label = QLabel('中位数差值百分比: -')
        left_panel.addWidget(self.top48_median_label)
        left_panel.addWidget(self.rest_median_label)
        left_panel.addWidget(self.difference_percentage_median_label)

        # Label management
        label_layout = QHBoxLayout()
        self.label_input = QLineEdit()
        self.add_label_btn = QPushButton('添加标签')
        self.add_label_btn.clicked.connect(self.add_label)
        label_layout.addWidget(self.label_input)
        label_layout.addWidget(self.add_label_btn)
        left_panel.addLayout(label_layout)

        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QListWidget.MultiSelection)  # 允许多选
        self.label_list.addItems(['左', '平', '右'])
        left_panel.addWidget(self.label_list)

        self.remove_label_btn = QPushButton('删除选中标签')
        self.remove_label_btn.clicked.connect(self.remove_label)
        left_panel.addWidget(self.remove_label_btn)

        # Data collection controls
        collection_layout = QHBoxLayout()
        self.start_btn = QPushButton('开始采集')
        self.start_btn.clicked.connect(self.start_collecting)
        self.pause_btn = QPushButton('暂停采集')
        self.pause_btn.clicked.connect(self.pause_collecting)
        self.stop_btn = QPushButton('结束采集')
        self.stop_btn.clicked.connect(self.stop_collecting)
        collection_layout.addWidget(self.start_btn)
        collection_layout.addWidget(self.pause_btn)
        collection_layout.addWidget(self.stop_btn)
        left_panel.addLayout(collection_layout)

        # Playback controls
        playback_layout = QHBoxLayout()
        self.playback_btn = QPushButton('从文件回放')
        self.playback_btn.clicked.connect(self.start_playback_from_file)
        self.playback_slider = QSlider(Qt.Horizontal)
        self.playback_slider.setEnabled(False)
        self.playback_slider.valueChanged.connect(self.slider_value_changed)
        playback_layout.addWidget(self.playback_btn)
        playback_layout.addWidget(self.playback_slider)
        left_panel.addLayout(playback_layout)

        # Right panel for visualizations
        right_panel = QVBoxLayout()
        
        # Heatmap display
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        right_panel.addWidget(self.canvas)

        # Matrix display
        self.matrix_label = QLabel('矩阵预览:')
        right_panel.addWidget(self.matrix_label)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

        self.setLayout(main_layout)

    def add_label(self):
        new_label = self.label_input.text().strip()
        if new_label and new_label not in [self.label_list.item(i).text() for i in range(self.label_list.count())]:
            self.label_list.addItem(new_label)
            self.label_input.clear()
        else:
            QMessageBox.warning(self, '警告', '标签不能为空或重复')

    def remove_label(self):
        for item in self.label_list.selectedItems():
            self.label_list.takeItem(self.label_list.row(item))

    def start_collecting(self):
        if not self.collecting:
            selected_labels = [item.text() for item in self.label_list.selectedItems()]
            if not selected_labels:
                QMessageBox.warning(self, '警告', '请至少选择一个标签')
                return
            self.file_name = '_'.join(selected_labels)
            self.collecting = True
            self.paused = False
            self.collected_data = []
            self.start_collection.emit()
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)

    def pause_collecting(self):
        if self.collecting and not self.paused:
            self.paused = True
            self.pause_collection.emit()
            self.pause_btn.setText('继续采集')
        elif self.collecting and self.paused:
            self.paused = False
            self.start_collection.emit()
            self.pause_btn.setText('暂停采集')

    def stop_collecting(self):
        if self.collecting:
            self.collecting = False
            self.paused = False
            self.stop_collection.emit()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.pause_btn.setText('暂停采集')
            self.save_collected_data()

    def update_web_info(self, posture, confidence, timestamp):
        self.web_info_label.setText(f'当前睡姿: {posture}')
        self.confidence_label.setText(f'置信度: {confidence:.2f}')
        self.timestamp_label.setText(f'最后更新时间: {timestamp}')

    def update_metrics(self, top48_avg, rest_avg, difference_percentage, top48_median, rest_median, difference_percentage_median):
        self.top48_avg_label.setText(f'Top48均值: {top48_avg:.2f}')
        self.rest_avg_label.setText(f'其余均值: {rest_avg:.2f}')
        self.difference_percentage_label.setText(f'差值百分比: {difference_percentage:.2f}%')
        self.top48_median_label.setText(f'Top48中位数: {top48_median:.2f}')
        self.rest_median_label.setText(f'其余中位数: {rest_median:.2f}')
        self.difference_percentage_median_label.setText(f'中位数差值百分比: {difference_percentage_median:.2f}%')

    def save_collected_data(self):
        if self.collected_data:
            # 创建Collect_data文件夹（如果不存在）
            collect_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Collect_data')
            os.makedirs(collect_data_dir, exist_ok=True)

            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"{self.file_name}_{timestamp}.csv"
            default_path = os.path.join(collect_data_dir, default_filename)

            file_path, _ = QFileDialog.getSaveFileName(self, '保存数据', default_path, 'CSV Files (*.csv)')
            if file_path:
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['timestamp', 'data'])
                    for timestamp, matrix_str in self.collected_data:
                        writer.writerow([timestamp, matrix_str])
                QMessageBox.information(self, '保存成功', f'数据已保存到 {file_path}')
                self.playback_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, '警告', '没有可保存的数据')

    def start_playback_from_file(self):
        collect_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Collect_data')
        file_path, _ = QFileDialog.getOpenFileName(self, '选择CSV文件', collect_data_dir, 'CSV Files (*.csv)')
        if file_path:
            self.playback_data = []
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # 跳过标题行
                for row in reader:
                    timestamp, matrix_str = row
                    self.playback_data.append((timestamp, matrix_str))

            if self.playback_data:
                self.playback_index = 0
                self.playback_slider.setRange(0, len(self.playback_data) - 1)
                self.playback_slider.setValue(0)
                self.playback_slider.setEnabled(True)

                if self.playback_window is None:
                    self.playback_window = PlaybackWindow(self)
                self.playback_window.show()

                self.update_playback()
                self.playback_timer.start(500)  # Update every 500 ms
            else:
                QMessageBox.warning(self, '警告', '选择的文件没有有效数据')

    def update_playback(self):
        if self.playback_index < len(self.playback_data):
            timestamp, matrix_str = self.playback_data[self.playback_index]
            matrix = np.array(eval(matrix_str)).reshape((16, 10))
            self.playback_window.update_heatmap(matrix)
            self.update_matrix_display(matrix)
            self.playback_index += 1
            self.playback_slider.setValue(self.playback_index)
        else:
            self.playback_timer.stop()

    def slider_value_changed(self):
        self.playback_index = self.playback_slider.value()
        if self.playback_index < len(self.playback_data):
            timestamp, matrix_str = self.playback_data[self.playback_index]
            matrix = np.array(eval(matrix_str)).reshape((16, 10))
            self.playback_window.update_heatmap(matrix)
            self.update_matrix_display(matrix)

    def update_matrix_display(self, matrix):
        matrix_str = '\n'.join([' '.join(f'{x:3d}' for x in row) for row in matrix])
        self.matrix_label.setText(f'矩阵预览:\n{matrix_str}')

    def update_heatmap(self, matrix, top_points, centroid, direction_vector, angle, timestamp):
        self.ax.clear()
        
        # Draw the complete heatmap
        cax = self.ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect=1.2)
        
        # 只在第一次创建颜色条或者颜色条不存在时创建
        if self.heatmap_colorbar is None:
            self.heatmap_colorbar = self.figure.colorbar(cax, ax=self.ax, label='压力值')
        else:
            self.heatmap_colorbar.update_normal(cax)

        if top_points is not None and centroid is not None and direction_vector is not None:
            # Draw the main direction line
            scale = max(matrix.shape) / 2
            start_point = centroid - scale * direction_vector
            end_point = centroid + scale * direction_vector
            self.ax.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], 'r-', linewidth=2, label='主方向')

            # Highlight the selected points
            self.ax.scatter(top_points[:, 1], top_points[:, 0], color='white', edgecolor='black', s=50, label='选中点')

            # Highlight the centroid
            self.ax.scatter(centroid[1], centroid[0], color='green', edgecolor='black', s=100, label='加权质心')

            # Calculate and display the angle
            angle_text = f"角度: {angle:.2f}°"
            self.ax.text(0.05, 0.95, angle_text, transform=self.ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        self.ax.set_title("睡姿热力图")
        self.ax.set_xlabel("X轴")
        self.ax.set_ylabel("Y轴")
        self.ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

        self.canvas.draw()

    def collect_raw_matrix(self, matrix):
        if self.collecting and not self.paused:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            matrix_str = ','.join(map(str, matrix.flatten()))
            self.collected_data.append((timestamp, matrix_str))

def run_gui():
    ex = DataCollectionGUI()
    return ex

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = run_gui()
    sys.exit(app.exec_())