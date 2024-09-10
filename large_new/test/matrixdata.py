import numpy as np
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QRectF, QTimer, QObject
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush
import sys
import threading
import queue
import warnings

# 忽略特定的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

def adc_to_pressure_matrix(adc_matrix, vdd=3.3, r=10000, a=1000, b=5000, k=1023/3.3):
    def adc_to_pressure(adc):
        vo = np.maximum(adc / k, 1e-10)  # 避免除以零
        rsens = r * (vdd / vo - 1)
        return np.clip(np.exp((rsens - b) / a), 0, 1e6)  # 限制压力值范围

    # 将 ADC 矩阵转换为压力矩阵
    pressure_matrix = adc_to_pressure(adc_matrix)

    # 应用校正
    corrected_pressure = correct_pressure_distribution(pressure_matrix)

    return corrected_pressure

def correct_pressure_distribution(pressure_matrix):
    # 处理无效值
    pressure_matrix = np.nan_to_num(pressure_matrix, nan=0.0, posinf=1e6, neginf=0.0)

    # 1. 去除异常值
    mean = np.mean(pressure_matrix)
    std = np.std(pressure_matrix)
    pressure_matrix = np.clip(pressure_matrix, mean - 3*std, mean + 3*std)

    # 2. 应用平滑滤波
    from scipy.ndimage import gaussian_filter
    smoothed_matrix = gaussian_filter(pressure_matrix, sigma=1)

    # 3. 均衡化处理
    row_sums = smoothed_matrix.sum(axis=1, keepdims=True)
    col_sums = smoothed_matrix.sum(axis=0, keepdims=True)
    total_sum = smoothed_matrix.sum()

    row_factors = np.sqrt(np.maximum(total_sum / (32 * row_sums), 1e-10))
    col_factors = np.sqrt(np.maximum(total_sum / (32 * col_sums), 1e-10))

    corrected_pressure = smoothed_matrix * row_factors * col_factors

    # 4. 归一化处理
    min_val = np.min(corrected_pressure)
    max_val = np.max(corrected_pressure)
    if max_val > min_val:
        corrected_pressure = (corrected_pressure - min_val) / (max_val - min_val)
    else:
        corrected_pressure = np.zeros_like(corrected_pressure)

    return corrected_pressure

def setup_serial_port():
    available_ports = list(serial.tools.list_ports.comports())
    for port in available_ports:
        try:
            s = serial.Serial(port.device, 1000000, timeout=1)
            print(f'成功连接到串口: {port.device}')
            return s
        except:
            print(f'无法连接到串口: {port.device}, 尝试下一个...')
    print('未找到可用的串口设备')
    return None

def find_packet_start(data):
    return np.where(np.all(np.array([data[i:i+4] for i in range(len(data)-3)]) == [170, 85, 3, 153], axis=1))[0]

def read_matrix_from_serial(ser):
    global alld
    if ser.in_waiting > 0:
        receive = ser.read(ser.in_waiting)
        alld = np.concatenate([alld, np.frombuffer(receive, dtype=np.uint8)])
        
        if len(alld) >= 4:
            index = find_packet_start(alld)
            if len(index) > 0:
                if index[0] > 0:
                    alld = alld[index[0]:]
                
                if len(alld) >= 1028:
                    imgdata = alld[4:1028]
                    alld = alld[1028:]
                    
                    if len(imgdata) == 1024:
                        img_data = np.array(imgdata).flatten()
                        
                        for i in range(8):
                            start1, end1 = i * 32, (i + 1) * 32
                            start2, end2 = (14 - i) * 32, (15 - i) * 32
                            img_data[start1:end1], img_data[start2:end2] = img_data[start2:end2].copy(), img_data[start1:end1].copy()
                        
                        img_data = np.roll(img_data, -15 * 32)
                        img_data = img_data.reshape(32, 32)
                        return img_data

    return None

class MatrixDisplay(QWidget):
    update_matrix_signal = pyqtSignal(np.ndarray)
    update_mean_signal = pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.ser = setup_serial_port()
        self.update_matrix_signal.connect(self.update_matrix)
        self.update_mean_signal.connect(self.update_mean_labels)
        self.color_cache = {}
        self.selection_start = None
        self.selection_end = None
        self.current_matrix = None
        self.setMouseTracking(True)
        self.last_valid_ratio = None
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # 更新频率设置为100ms

    def initUI(self):
        self.setWindowTitle('矩阵显示')
        self.showMaximized()  # 全屏显示

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.matrix_layout = self.create_matrix_layout()
        main_layout.addLayout(self.matrix_layout)

        mean_layout = QHBoxLayout()
        self.selected_mean_label = QLabel("选中区域均值: N/A")
        self.total_mean_label = QLabel("总体均值: N/A")
        self.bed_exit_indicator = QLabel("离床指标: N/A")
        mean_layout.addWidget(self.selected_mean_label)
        mean_layout.addWidget(self.total_mean_label)
        mean_layout.addWidget(self.bed_exit_indicator)
        main_layout.addLayout(mean_layout)

    def create_matrix_layout(self):
        layout = QGridLayout()
        layout.setSpacing(1)  # 减小格子间距

        font = QFont("Courier")
        font.setPointSize(8)  # 减小字体大小以适应更小的格子

        self.matrix_labels = [[QLabel('0') for _ in range(32)] for _ in range(32)]

        for i in range(32):
            for j in range(32):
                label = self.matrix_labels[i][j]
                label.setFont(font)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #e0e0e0;")
                layout.addWidget(label, i, j)

        return layout

    def update_display(self):
        if self.current_matrix is not None:
            for i in range(32):
                for j in range(32):
                    value = int(self.current_matrix[i, j])
                    item = self.matrix_labels[i][j]
                    if item.text() != str(value):
                        item.setText(str(value))
                        self.set_color(item, value)

            pooled_matrix = self.pool_matrix(self.current_matrix, (2, 3))
            ratio = self.calculate_harmonic_mean(pooled_matrix)
            
            if ratio > 0 or self.last_valid_ratio is None:
                self.last_valid_ratio = ratio
            
            selected_mean = self.calculate_selected_mean()
            
            self.update_mean_signal.emit(selected_mean, np.mean(self.current_matrix), self.last_valid_ratio)

    def set_color(self, label, value):
        value = max(0, min(255, value))
        if value not in self.color_cache:
            intensity = value / 255
            bg_color = QColor(255 - int(200 * intensity), 255 - int(200 * intensity), 255)
            self.color_cache[value] = f"background-color: {bg_color.name()};"
        
        label.setStyleSheet(self.color_cache[value])

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_start = self.get_matrix_position(event.pos())
            self.selection_end = None

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.selection_end = self.get_matrix_position(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_end = self.get_matrix_position(event.pos())
            self.update()

    def get_matrix_position(self, pos):
        for i in range(32):
            for j in range(32):
                if self.matrix_labels[i][j].geometry().contains(pos):
                    return i, j
        return None

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_start and self.selection_end:
            painter = QPainter(self)
            
            # 设置选择框的颜色和样式
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            
            # 计算选择区域
            start_row, start_col = min(self.selection_start[0], self.selection_end[0]), min(self.selection_start[1], self.selection_end[1])
            end_row, end_col = max(self.selection_start[0], self.selection_end[0]), max(self.selection_start[1], self.selection_end[1])
            start_item = self.matrix_labels[start_row][start_col]
            end_item = self.matrix_labels[end_row][end_col]
            rect = QRectF(start_item.geometry().topLeft(), end_item.geometry().bottomRight())
            
            # 绘制透明红色背景
            painter.setBrush(QBrush(QColor(255, 0, 0, 50)))  # 50 是透明度，可以调整
            painter.drawRect(rect)
            
            # 绘制红色边框
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

    def calculate_selected_mean(self):
        if self.current_matrix is None or self.selection_start is None or self.selection_end is None:
            return 0
        
        start_row, start_col = min(self.selection_start[0], self.selection_end[0]), min(self.selection_start[1], self.selection_end[1])
        end_row, end_col = max(self.selection_start[0], self.selection_end[0]), max(self.selection_start[1], self.selection_end[1])
        selected_area = self.current_matrix[start_row:end_row+1, start_col:end_col+1]
        return np.mean(selected_area)

    def pool_matrix(self, matrix, pool_size):
        """
        对矩阵进行池化操作
        """
        m, n = pool_size
        h, w = matrix.shape
        
        new_h = (h // m) * m
        new_w = (w // n) * n
        
        crop_h = (h - new_h) // 2
        crop_w = (w - new_w) // 2
        
        matrix = matrix[crop_h:crop_h + new_h, crop_w:crop_w + new_w]
        
        pooled_matrix = matrix.reshape(new_h // m, m, new_w // n, n).mean(axis=(1, 3))
        
        return pooled_matrix

    def calculate_harmonic_mean(self, matrix):
        """
        计算调和平均数作为离床指标
        """
        sorted_values = np.sort(matrix.flatten())[::-1]
        top_16_median = np.mean(sorted_values[:16])
        top_32_median = np.mean(sorted_values[:48])
        
        if top_16_median + top_32_median > 0:
            harmonic_mean = 2 * (top_16_median * top_32_median) / (top_16_median + top_32_median)
        else:
            harmonic_mean = 0
        
        return harmonic_mean / 255

    @pyqtSlot(float, float, float)
    def update_mean_labels(self, selected_mean, total_mean, ratio):
        self.selected_mean_label.setText(f"选中区域均值: {selected_mean:.2f}")
        self.total_mean_label.setText(f"总体均值: {total_mean:.2f}")
        self.bed_exit_indicator.setText(f"离床指标: {ratio:.4f}")

    @pyqtSlot(np.ndarray)
    def update_matrix(self, matrix):
        self.current_matrix = matrix

class SerialReader(threading.Thread):
    def __init__(self, serial_port, data_queue):
        threading.Thread.__init__(self)
        self.serial_port = serial_port
        self.data_queue = data_queue
        self.running = True

    def run(self):
        global alld
        while self.running:
            matrix = read_matrix_from_serial(self.serial_port)
            if matrix is not None:
                self.data_queue.put(matrix)

    def stop(self):
        self.running = False

class DisplayUpdater(QObject):
    def __init__(self, data_queue, matrix_display):
        super().__init__()
        self.data_queue = data_queue
        self.matrix_display = matrix_display
        self.running = True

    def run(self):
        while self.running:
            try:
                matrix = self.data_queue.get(timeout=1)
                self.matrix_display.update_matrix_signal.emit(matrix)
            except queue.Empty:
                pass

    def stop(self):
        self.running = False

if __name__ == '__main__':
    global alld
    alld = np.array([], dtype=np.uint8)
    
    app = QApplication(sys.argv)
    ex = MatrixDisplay()
    
    data_queue = queue.Queue()
    serial_reader = SerialReader(ex.ser, data_queue)
    display_updater = DisplayUpdater(data_queue, ex)
    
    serial_reader_thread = threading.Thread(target=serial_reader.run)
    display_updater_thread = threading.Thread(target=display_updater.run)
    
    serial_reader_thread.start()
    display_updater_thread.start()
    
    app.exec_()
    
    serial_reader.stop()
    display_updater.stop()
    serial_reader_thread.join()
    display_updater_thread.join()