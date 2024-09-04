import sys
import csv
import numpy as np
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QListWidget, QLineEdit, 
                             QMessageBox, QFileDialog, QSlider, QSplitter,
                             QDialog, QVBoxLayout, QFrame, QScrollArea)
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QResizeEvent
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import deque

# 全局变量
matrix_buffer = deque(maxlen=100)  # 存储最近100帧矩阵数据

class QCollapsibleFrame(QFrame):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(title)
        self.toggle_button.setStyleSheet("text-align: left; padding: 5px;")
        self.toggle_button.clicked.connect(self.toggle_collapse)

        self.content_area = QScrollArea()
        self.content_area.setWidgetResizable(True)
        self.content_area.setVisible(False)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_area.setWidget(self.content_widget)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

    def toggle_collapse(self):
        self.content_area.setVisible(not self.content_area.isVisible())

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

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
        self.ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect=1.5)
        self.ax.set_title("回放热力图")
        self.ax.set_xlabel("X轴")
        self.ax.set_ylabel("Y轴")
        self.canvas.draw()

class DataCollectionGUI(QWidget):
    start_collection = pyqtSignal()
    pause_collection = pyqtSignal()
    stop_collection = pyqtSignal()

    def __init__(self):
        global matrix_buffer
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
        self.matrix_count = 0

        # GUI更新定时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_gui)
        self.update_timer.start(100)  # 每100毫秒更新一次GUI

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

        # Matrix metrics display
        metrics_frame = QCollapsibleFrame("矩阵指标")
        self.bed_status_label = QLabel('床上状态: - (置信度: -%)')
        self.edge_status_label = QLabel('边缘状态: - (置信度: -%)')
        self.centroid_label = QLabel('质心坐标: -')
        self.rest_avg_label = QLabel('其余均值: -')
        self.top48_median_label = QLabel('Top48中位数: -')
        self.rest_median_label = QLabel('其余中位数: -')
        
        metrics_frame.add_widget(self.bed_status_label)
        metrics_frame.add_widget(self.edge_status_label)
        metrics_frame.add_widget(self.centroid_label)
        metrics_frame.add_widget(self.rest_avg_label)
        metrics_frame.add_widget(self.top48_median_label)
        metrics_frame.add_widget(self.rest_median_label)
        
        left_panel.addWidget(metrics_frame)

        # Label management
        label_layout = QHBoxLayout()
        self.label_input = QLineEdit()
        self.add_label_btn = QPushButton('添加标签')
        self.add_label_btn.clicked.connect(self.add_label)
        label_layout.addWidget(self.label_input)
        label_layout.addWidget(self.add_label_btn)
        left_panel.addLayout(label_layout)

        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QListWidget.MultiSelection)
        self.label_list.addItems(['左', '平', '右'])
        left_panel.addWidget(self.label_list)

        self.remove_label_btn = QPushButton('删除选中标签')
        self.remove_label_btn.clicked.connect(self.remove_label)
        left_panel.addWidget(self.remove_label_btn)

        # 在数据采集控件之前添加显示矩阵个数的标签
        self.matrix_count_label = QLabel('采集到的矩阵个数: 0')
        left_panel.addWidget(self.matrix_count_label)

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

        # Matrix display (collapsible)
        self.matrix_frame = QCollapsibleFrame("矩阵预览")
        self.matrix_label = QLabel('矩阵数据将在这里显示')
        self.matrix_label.setFont(QFont('Courier', 8))  # 使用等宽字体
        self.matrix_frame.add_widget(self.matrix_label)
        right_panel.addWidget(self.matrix_frame)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)

        self.setLayout(main_layout)

    def receive_new_data(self, matrix):
        global matrix_buffer
        matrix_buffer.append(matrix)
        # print(f"Received new data. Matrix buffer size: {len(matrix_buffer)}")
        if self.collecting and not self.paused:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            matrix_str = ','.join(map(str, matrix.flatten()))
            self.collected_data.append((timestamp, matrix_str))
            self.matrix_count += 1
            self.update_gui()

    def update_gui(self):
        self.update_matrix_count_label()
        # print(f"GUI updated. Matrices collected: {self.matrix_count}")

    def update_matrix_count_label(self):
        self.matrix_count_label.setText(f'采集到的矩阵个数: {self.matrix_count}')

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
            if not self.collected_data:  # 只有在没有数据时才重置
                self.collected_data = []
                self.matrix_count = 0
            self.start_collection.emit()
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.pause_btn.setText('暂停采集')
            self.update_matrix_count_label()
            print("Data collection started")

    def pause_collecting(self):
        if self.collecting:
            if not self.paused:
                self.paused = True
                self.pause_collection.emit()
                self.pause_btn.setText('继续采集')
            else:
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
            self.update_matrix_count_label()

    def update_web_info(self, posture, confidence, timestamp):
        # print(f"Updating web info: {posture}, {confidence}, {timestamp}")
        self.web_info_label.setText(f'当前睡姿: {posture}')
        self.confidence_label.setText(f'置信度: {confidence:.2f}')
        self.timestamp_label.setText(f'最后更新时间: {timestamp}')

    def update_metrics(self, bed_status, edge_status, centroid,  rest_avg, top48_median, rest_median):
        # if bed_status[1] < 0.15:
        #     posture = '离床'
        #     confidence = 1.0
        # else:
        #     posture = bed_status[0]
        #     confidence = bed_status[1]
        # self.web_info_label.setText(f'当前睡姿: {posture}')
        # self.confidence_label.setText(f'置信度: {confidence:.2f}')
        self.bed_status_label.setText(f'床上状态: {bed_status[0]} (计算比例: {bed_status[1]:.2f}%)')
        self.edge_status_label.setText(f'边缘状态: {edge_status[0]} (置信度: {edge_status[1]:.2f}%)')
        self.centroid_label.setText(f'质心坐标：({centroid[0]:.2f}, {centroid[1]:.2f})')
        self.rest_avg_label.setText(f'其余均值: {rest_avg:.2f}')
        self.top48_median_label.setText(f'Top48中位数: {top48_median:.2f}')
        self.rest_median_label.setText(f'其余中位数: {rest_median:.2f}')
    
    def save_collected_data(self):
        if self.collected_data:
            print(f"Data to be saved: {len(self.collected_data)} entries")
            collect_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Collect_data')
            os.makedirs(collect_data_dir, exist_ok=True)

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
                
                # 重置采集状态
                self.collected_data = []
                self.matrix_count = 0
                self.update_matrix_count_label()
        else:
            print("No data collected to save")
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
            matrix = np.array(eval(matrix_str)).reshape((32, 32))
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
            matrix = np.array(eval(matrix_str)).reshape((32, 32))
            self.playback_window.update_heatmap(matrix)
            self.update_matrix_display(matrix)

    def update_matrix_display(self, matrix):
        matrix_str = '\n'.join([' '.join(f'{x:3d}' for x in row) for row in matrix])
        self.matrix_label.setText(f'矩阵预览:\n{matrix_str}')
        # 确保矩阵预览面板是展开的
        self.matrix_frame.content_area.setVisible(True)

    def update_heatmap(self, matrix, top_points, centroid, direction_vector, angle, timestamp):
        # print(f"Updating heatmap in GUI")
        self.ax.clear()
        height, width = matrix.shape
        # 绘制完整的热力图
        cax = self.ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect=1.2)

        # 只在第一次创建颜色条或者颜色条不存在时创建
        if self.heatmap_colorbar is None:
            self.heatmap_colorbar = self.figure.colorbar(cax, ax=self.ax, label='压力值')
        else:
            self.heatmap_colorbar.update_normal(cax)

        
        # 设置刻度标签，使其从1开始
        self.ax.set_xticks(range(width))
        self.ax.set_yticks(range(height))
        self.ax.set_xticklabels(range(1, width + 1))
        self.ax.set_yticklabels(range(1, height + 1))
        
        if top_points is not None and centroid is not None and direction_vector is not None:
            # 裁剪线段以确保不会超出热力图范围
            height, width = matrix.shape

            def clip_line_to_bounds(start, end, width, height):
                def clip(p, pmin, pmax):
                    return max(min(p, pmax), pmin)

                def compute_intersection(p1, p2, axis, value):
                    delta = p2 - p1
                    t = (value - p1[axis]) / delta[axis]
                    return p1 + t * delta

                # 裁剪X轴
                if start[1] < 0:
                    start = compute_intersection(start, end, 1, 0)
                if end[1] < 0:
                    end = compute_intersection(start, end, 1, 0)
                if start[1] > width:
                    start = compute_intersection(start, end, 1, width)
                if end[1] > width:
                    end = compute_intersection(start, end, 1, width)

                # 裁剪Y轴
                if start[0] < 0:
                    start = compute_intersection(start, end, 0, 0)
                if end[0] < 0:
                    end = compute_intersection(start, end, 0, 0)
                if start[0] > height:
                    start = compute_intersection(start, end, 0, height)
                if end[0] > height:
                    end = compute_intersection(start, end, 0, height)

                return start, end

            scale = max(matrix.shape) / 2
            start_point = centroid - scale * direction_vector
            end_point = centroid + scale * direction_vector
            start_point, end_point = clip_line_to_bounds(start_point, end_point, width-1, height-1)

            # 绘制主方向线
            self.ax.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], 'r-', linewidth=2, label='主方向')

            # 突出显示选中的点
            self.ax.scatter(top_points[:, 1], top_points[:, 0], color='white', edgecolor='white', s=5, label='选中点')

            # 突出显示加权质心
            self.ax.scatter(centroid[1], centroid[0], color='green', edgecolor='black', s=100, label='加权质心')

            # 计算并显示角度
            angle_text = f"角度: {angle:.2f}°"
            self.ax.text(0.05, 0.95, angle_text, transform=self.ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        # 调整图例位置，放在热力图右上角
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small')

        # 调整布局以确保图例和颜色条都不遮挡热力图
        self.figure.tight_layout()

        self.ax.set_title("睡姿热力图")
        self.ax.set_xlabel("X轴")
        self.ax.set_ylabel("Y轴")

        self.canvas.draw()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        # 调整矩阵预览的宽度以匹配热力图的宽高比
        heatmap_width = self.canvas.width()
        heatmap_height = self.canvas.height()
        if heatmap_width > 0 and heatmap_height > 0:
            aspect_ratio = 1.2  # 与热力图相同的宽高比
            matrix_width = int(heatmap_height * aspect_ratio)
            self.matrix_frame.setFixedWidth(min(matrix_width, heatmap_width))

def run_gui():
    ex = DataCollectionGUI()
    return ex

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = run_gui()
    sys.exit(app.exec_())