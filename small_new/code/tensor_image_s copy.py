import serial  # 用于串口通信
import serial.tools.list_ports  # 用于列出所有可用的串口
import numpy as np  # 用于进行数组和矩阵操作
import time  # 提供时间相关的功能
import logging  # 用于记录日志信息
import tensorflow as tf  # TensorFlow用于机器学习和深度学习操作
from flask import Flask, render_template_string, jsonify, send_file  # Flask是一个轻量级的Web框架，用于创建Web应用
from threading import Thread, Event, Lock  # 提供线程、事件和锁的支持，用于多线程编程
import socket  # 提供底层网络接口
import netifaces  # 用于获取网络接口信息
import signal  # 提供对操作系统信号的支持
import sys  # 提供对Python解释器的访问
import atexit  # 提供程序退出时执行清理操作的功能
import matplotlib  # 用于数据可视化
matplotlib.use('Agg')  # 设置Matplotlib为非交互式后端，用于在后台生成图像
import matplotlib.pyplot as plt  # 导入Matplotlib的绘图接口
from sklearn.decomposition import PCA  # 导入主成分分析 (PCA) 模块，用于数据降维
import io  # 提供操作流的工具，如内存中的文件
from queue import Queue  # 提供线程安全的队列
from collections import deque  # 提供双端队列，用于高效的插入和删除操作
from queue import Queue, Empty
from PyQt5.QtWidgets import QApplication  # 用于创建PyQt5应用程序的入口
from PyQt5.QtCore import QThread, pyqtSignal, QTimer  # 提供线程、信号槽、定时器等功能
from PyQt5.QtGui import QFont  # 提供字体支持
from gui_module import run_gui  # 导入自定义的GUI模块，确保该模块在同一目录下
from movement_detection import DataCollector, DataVisualizer
import io
from PIL import Image

# 设置Matplotlib的中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置SimHei字体用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决Matplotlib无法正常显示负号的问题

# 设置日志格式和级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 全局变量定义
global matrix_buffer, latest_prediction, latest_heatmap, heatmap_timestamp, running, exit_event, ser, alld, posture_labels
matrix_buffer = deque(maxlen=100)  # 存储最近100帧的矩阵数据，用于推理
latest_prediction = {"posture": "未知", "confidence": 0.0, "timestamp": time.time()}  # 用于存储最新的预测结果
latest_heatmap = None  # 用于存储最新的热力图图像
heatmap_timestamp = 0  # 用于记录热力图的更新时间戳
heatmap_lock = Lock()  # 用于在多线程环境下保护热力图的访问
running = True  # 标识程序的运行状态
exit_event = Event()  # 用于控制程序的退出
ser = None  # 串口对象初始化为空
alld = np.array([], dtype=np.uint8)  # 初始化空的数组，用于接收串口数据
posture_labels = ['平躺', '左侧卧', '右侧卧']  # 定义睡姿的标签

# 创建热力图的Figure和Axes对象（在主线程中创建）
heatmap_fig, heatmap_ax = plt.subplots(figsize=(10, 8))  # 创建一个10x8英寸的画布和子图
heatmap_colorbar = None  # 初始化热力图的颜色条为None

inference_results = Queue(maxsize=1)  # 存储最新的推理结果，队列大小为1
metrics_results = Queue(maxsize=1)  # 存储最新的指标计算结果，队列大小为1

# 加载训练好的TensorFlow模型
model = tf.saved_model.load(r'D:\repository\deeplearning\small_new\Data\tensorflow\ncz\1')
logging.info("模型加载成功")  # 打印模型加载成功的日志信息


# Flask应用初始化
app = Flask(__name__)  # 创建Flask应用

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.label = QLabel()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def update_plot(self, plot_image):
        height, width, channel = plot_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(plot_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
from queue import Queue

class MovementDetectionThread(QThread):
    update_movement_plot = pyqtSignal(object)

    def __init__(self, parent=None, show_local_plot=False):
        super().__init__(parent)
        self.data_collector = DataCollector(ylimnum=10)
        self.data_visualizer = DataVisualizer(self.data_collector)
        self.running = True
        self.data_queue = Queue()
        self.show_local_plot = show_local_plot
        self.local_plot_window = None
        self.last_update_time = 0

        if self.show_local_plot:
            self.setup_local_plot_window()

    def setup_local_plot_window(self):
        self.local_plot_window = QWidget()
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.local_plot_window.setLayout(layout)
        self.local_plot_window.setWindowTitle("体动检测图")
        self.local_plot_window.show()

    def run(self):
        while self.running:
            if not self.data_queue.empty():
                # 处理队列中的所有新数据
                while not self.data_queue.empty():
                    new_matrix = self.data_queue.get()
                    self.data_collector.process_matrix(new_matrix)
                
                current_time = time.time()
                if current_time - self.last_update_time >= 1:  # 每秒更新一次
                    # 生成并发送图像更新
                    plot_image = self.data_visualizer.get_plot_image()
                    self.update_movement_plot.emit(plot_image)

                    if self.show_local_plot and self.local_plot_window:
                        self.update_local_plot(plot_image)

                    self.last_update_time = current_time
            
            # 短暂休眠以避免过度消耗CPU
            time.sleep(0.01)

    def update_local_plot(self, plot_image):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(plot_image)
        self.canvas.draw()

    def stop(self):
        self.running = False
        if self.local_plot_window:
            self.local_plot_window.close()
        self.wait()

    def add_new_data(self, new_data):
        self.data_queue.put(new_data)

    def toggle_local_plot(self, show):
        self.show_local_plot = show
        if show and not self.local_plot_window:
            self.setup_local_plot_window()
        elif not show and self.local_plot_window:
            self.local_plot_window.close()
            self.local_plot_window = None

class MatrixMetrics:
    def __init__(self):
        self.bed_status = ("", 0)  # 在床/离床状态
        self.edge_status = ("", 0)  # 坠床/坐床边状态
        self.centroid = (0, 0)  # 重心坐标
        self.rest_avg = 0
        self.top48_median = 0
        self.rest_median = 0

    def calculate(self, matrix):
        # 计算在床/离床状态
        ratio = self.calculate_harmonic_mean(matrix)
        self.bed_status = ("在床" if ratio > 0.08 else "离床", ratio * 100)

        # 计算坠床/坐床边状态
        centroid = self.calculate_weighted_centroid(matrix)
        #赋值给self.centroid,转化为元组,横纵坐标+1
        self.centroid = tuple(map(lambda x: x+1, centroid))
        
        features = self.extract_features(matrix)
        probability = self.embedded_system_logic(features)
        
        if centroid[1] < 3 or centroid[1] > 7:
            if probability > 0.5:
                status = "坠床风险"
            else:
                status = "坐床边"
            self.edge_status = (status, max(probability, 1 - probability) * 100)
        elif centroid[1] < 4 or centroid[1] >= 6:
            if probability < 0.5:
                status = "坐床边"
                self.edge_status = (status, max(probability, 1 - probability) * 100)
            else:
                status = "正常"
                self.edge_status = ("正常", 100)
        else:
            status = "正常"
            self.edge_status = ("正常", 100)
        

        # 计算其他指标
        flat_matrix = matrix.flatten()
        sorted_matrix = np.sort(flat_matrix)[::-1]

        # self.top48_avg = np.mean(sorted_matrix[:48]) if len(sorted_matrix) >= 48 else 0
        self.rest_avg = np.mean(sorted_matrix[48:]) if len(sorted_matrix) > 48 else 0
        
        self.top48_median = np.median(sorted_matrix[:48]) if len(sorted_matrix) >= 48 else 0
        rest_elements = sorted_matrix[48:]
        non_zero_elements = rest_elements[rest_elements > 5]
        self.rest_median = np.mean(non_zero_elements) if len(non_zero_elements) > 0 else 5

    def calculate_harmonic_mean(self, matrix):
        sorted_values = np.sort(matrix.flatten())[::-1]
        top_16_median = np.mean(sorted_values[:16])
        top_32_median = np.mean(sorted_values[:32])
        
        if top_16_median + top_32_median > 0:
            harmonic_mean = 2 * (top_16_median * top_32_median) / (top_16_median + top_32_median)
        else:
            harmonic_mean = 0
        
        return harmonic_mean / 255

    def calculate_weighted_centroid(self, matrix, top_n=64):
        top_n = 20
        reshaped_matrix = matrix.reshape(16, 10)
        flat_indices = np.argsort(reshaped_matrix.flatten())[-top_n:]
        top_points = np.array(np.unravel_index(flat_indices, reshaped_matrix.shape)).T
        point_values = reshaped_matrix[top_points[:, 0], top_points[:, 1]]
        total_weight = np.sum(point_values)
        centroid = np.sum(top_points * point_values[:, np.newaxis], axis=0) / total_weight
        return centroid

    def extract_features(self, matrix):
        flat_matrix = matrix.flatten()
        
        top64_indices = np.argsort(flat_matrix)[-32:]
        top64_values = flat_matrix[top64_indices]
        valid_top64_indices = top64_indices[top64_values > 10]
        valid_top64_rows = valid_top64_indices // matrix.shape[1]
        unique_valid_top64_rows = np.unique(valid_top64_rows)
        
        min_val, max_val = np.min(flat_matrix), np.max(flat_matrix)
        threshold = min_val + 0.5 * (max_val - min_val)
        
        above_threshold_indices = np.where((flat_matrix >= threshold) & (flat_matrix > 10))[0]
        above_threshold_rows = above_threshold_indices // matrix.shape[1]
        unique_above_threshold_rows = np.unique(above_threshold_rows)
        
        return [len(unique_valid_top64_rows), len(unique_above_threshold_rows)]

    def embedded_system_logic(self, features):
        coefficients = np.array([2.2440458, 1.79786801])
        intercept = -29.922462671061126
        log_odds = np.dot(features, coefficients) + intercept
        return 1 / (1 + np.exp(-log_odds))

    def get_metrics(self):
        return {
            "bed_status": self.bed_status,
            "edge_status": self.edge_status,
            "centroid": self.centroid,
            "rest_avg": self.rest_avg,
            "top48_median": self.top48_median,
            "rest_median": self.rest_median
        }     
def get_ip_addresses():
    # 获取所有非回环网络接口的IP地址
    ip_addresses = []
    interfaces = netifaces.interfaces()  # 获取所有网络接口
    for interface in interfaces:
        if interface == 'lo':  # 跳过回环接口（lo）
            continue
        iface = netifaces.ifaddresses(interface).get(netifaces.AF_INET)  # 获取IPv4地址
        if iface:
            for addr in iface:
                ip_addresses.append(addr['addr'])  # 添加接口的IP地址到列表
    return ip_addresses  # 返回所有可用的IP地址

#flask路由主页
@app.route('/')
@app.route('/')
def home():
    # 处理主页的HTTP请求，返回一个HTML页面
    ip_addresses = get_ip_addresses()  # 获取服务器的IP地址列表
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>算法预测结果</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-color: #f0f0f0; }
                .container { max-width: 800px; margin: 0 auto; }
                .section { margin: 20px 0; background-color: #fff; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; }
                .section-header { padding: 10px; background-color: #f8f8f8; display: flex; justify-content: space-between; align-items: center; }
                .section-content { padding: 20px; max-height: 1000px; overflow: hidden; transition: max-height 0.3s ease-out; }
                .section-content.collapsed { max-height: 0; padding: 0; }
                #result { font-size: 24px; margin-bottom: 20px; color: #333; }
                #timestamp { font-size: 14px; color: #666; margin-bottom: 20px; }
                #metrics, #results { font-size: 16px; color: #333; text-align: left; }
                #heatmap { max-width: 100%; height: auto; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                #ip-addresses { font-size: 16px; margin-top: 20px; color: #333; }
                .toggle-btn { background-color: #4CAF50; border: none; color: white; padding: 5px 10px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; margin: 4px 2px; cursor: pointer; border-radius: 4px; }
                #movement-plot-container { max-height: 800px; overflow-y: auto; margin-top: 20px; }
                #movement-plot { width: 100%; height: auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            </style>
            <script>
                function toggleSection(sectionId) {
                    const content = document.getElementById(sectionId);
                    content.classList.toggle('collapsed');
                    const btn = content.previousElementSibling.querySelector('.toggle-btn');
                    btn.textContent = content.classList.contains('collapsed') ? '展开' : '折叠';
                }

                function updateResult() {
                    fetch('/get_latest')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('result').innerText = `当前睡姿: ${data.posture} (置信度: ${data.confidence.toFixed(2)})`;
                            document.getElementById('timestamp').innerText = `最后更新时间: ${new Date(data.timestamp * 1000).toLocaleString()}`;
                            document.getElementById('metrics').innerHTML = `
                                <p>床上状态: ${data.bed_status[0]} (计算比例: ${data.bed_status[1].toFixed(2)}%)</p>
                                <p>边缘状态: ${data.edge_status[0]} (置信度: ${data.edge_status[1].toFixed(2)}%)</p>
                                <p>重心坐标: (${data.centroid[0].toFixed(2)}, ${data.centroid[1].toFixed(2)})</p>
                                <p>其余均值: ${data.rest_avg.toFixed(2)}</p>
                                <p>Top48中位数: ${data.top48_median.toFixed(2)}</p>
                                <p>其余中位数: ${data.rest_median.toFixed(2)}</p>
                            `;
                            document.getElementById('heatmap').src = '/get_heatmap?' + new Date().getTime();
                        });
                }

                function updateMovementPlot() {
                    document.getElementById('movement-plot').src = '/get_movement_plot?' + new Date().getTime();
                }

                setInterval(updateResult, 1500);
                setInterval(updateMovementPlot, 1000);
            </script>
        </head>
        <body>
            <div class="container">
                <h1>算法预测结果</h1>
                <div class="section">
                    <div class="section-header">
                        <h2>数据指标</h2>
                        <button class="toggle-btn" onclick="toggleSection('metrics-content')">折叠</button>
                    </div>
                    <div id="metrics-content" class="section-content">
                        <div id="metrics"></div>
                    </div>
                </div>
                <div class="section">
                    <div class="section-header">
                        <h2>预测结果</h2>
                        <button class="toggle-btn" onclick="toggleSection('results-content')">折叠</button>
                    </div>
                    <div id="results-content" class="section-content">
                        <div id="result">加载中...</div>
                        <div id="timestamp"></div>
                        <img id="heatmap" src="/get_heatmap" alt="热力图">
                    </div>
                </div>
                <div class="section">
                    <div class="section-header">
                        <h2>体动检测图</h2>
                        <button class="toggle-btn" onclick="toggleSection('movement-plot-content')">折叠</button>
                    </div>
                    <div id="movement-plot-content" class="section-content">
                        <div id="movement-plot-container">
                            <img id="movement-plot" src="/get_movement_plot" alt="体动检测图">
                        </div>
                    </div>
                </div>
                <!-- 
                <div id="ip-addresses">
                    <p>可用的访问地址:</p>
                    {% for ip in ip_addresses %}
                        <p>http://{{ip}}:5000</p>
                    {% endfor %}
                </div>
                -->
            </div>
        </body>
        </html>
    ''', ip_addresses=ip_addresses)  # 使用Jinja模板引擎生成HTML并渲染页面

# 更新 Flask 路由
@app.route('/get_movement_plot')
def get_movement_plot():
    global movement_detection_thread
    plot_image = movement_detection_thread.data_visualizer.get_plot_image()
    img_io = io.BytesIO()
    Image.fromarray(plot_image).save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/get_latest')
def get_latest():
    global latest_prediction, heatmap_timestamp
    try:
        current_prediction = inference_results.get(block=False)
    except Empty:
        current_prediction = latest_prediction
    
    try:
        metrics = metrics_results.get(block=False)
    except Empty:
        metrics = {}
    
    return jsonify({
        **current_prediction, 
        "heatmap_timestamp": heatmap_timestamp, 
        "bed_status": metrics.get("bed_status", ("", 0)),
        "edge_status": metrics.get("edge_status", ("", 0)),
        "centroid": metrics.get("centroid", (0, 0)),
        "rest_avg": metrics.get("rest_avg", 0),
        "top48_median": metrics.get("top48_median", 0),
        "rest_median": metrics.get("rest_median", 0)
    })
    
@app.route('/get_heatmap')
def get_heatmap():
    # 处理获取最新热力图的HTTP请求
    global latest_heatmap
    with heatmap_lock:  # 使用锁来确保在多线程环境中对热力图的访问是安全的
        if latest_heatmap is None:
            return "No heatmap available", 404  # 如果没有热力图数据，返回404错误
        return send_file(io.BytesIO(latest_heatmap), mimetype='image/png')  # 将热力图数据作为PNG图像返回

def run_flask_app():
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)  
    # host='0.0.0.0' 表示服务器监听所有可用的网络接口
    # port=5000 指定应用运行的端口
    # debug=False 禁用调试模式
    # use_reloader=False 禁用代码变动时自动重启功能


def setup_serial_port():
    # 尝试连接到可用的串口设备
    available_ports = list(serial.tools.list_ports.comports())  # 获取所有可用的串口设备列表
    for port in available_ports:
        try:
            s = serial.Serial(port.device, 1000000, timeout=1)  # 尝试以1Mbps的波特率打开串口
            logging.info(f'成功连接到串口: {port.device}')  # 记录成功连接的日志信息
            return s  # 返回已连接的串口对象
        except:
            logging.warning(f'无法连接到串口: {port.device}, 尝试下一个...')  # 如果连接失败，记录警告信息并继续尝试下一个串口设备
    logging.error('未找到可用的串口设备')  # 如果没有可用的串口设备，记录错误信息
    return None  # 返回None表示没有可用的串口设备


def predict_posture(model, matrix):
    # 使用已加载的模型对输入矩阵进行姿态预测
    input_data = matrix.flatten().reshape(1, 160, 1)  # 将矩阵展开为一维数组，并调整形状为（1, 160, 1）
    input_data = tf.cast(input_data, tf.float32)  # 将数据类型转换为float32，以适应模型输入的要求
    
    predictions = model(input_data, training=False)  # 使用模型进行预测，禁用训练模式
    predicted_class = tf.argmax(predictions[0]).numpy()  # 获取预测结果中概率最大的类别索引
    confidence = tf.reduce_max(predictions[0]).numpy()  # 获取预测结果中最大概率值
    return predicted_class, confidence  # 返回预测的类别索引和置信度

def find_packet_start(data):
    # 寻找数据包的起始位置
    return np.where(np.all(np.array([data[i:i+4] for i in range(len(data)-3)]) == [170, 85, 3, 153], axis=1))[0]
    # 该函数通过查找固定的字节序列 [170, 85, 3, 153] 来确定数据包的起始位置
    # 它使用NumPy的where函数找到所有匹配起始标志的位置，并返回这些位置的索引

def read_matrix_from_serial(ser):
    global alld  # 使用全局变量alld来存储读取到的数据
    if ser.in_waiting > 0:
        receive = ser.read(ser.in_waiting)  # 读取串口中所有可用的数据
        alld = np.concatenate([alld, np.frombuffer(receive, dtype=np.uint8)])  # 将读取的数据添加到全局缓冲区alld
        
        if len(alld) >= 4:
            index = find_packet_start(alld)  # 寻找数据包的起始位置
            if len(index) > 0:
                if index[0] > 0:
                    alld = alld[index[0]:]  # 删除起始位置之前的数据，保留从起始位置开始的数据
                
                if len(alld) >= 1028:
                    imgdata = alld[4:1028]  # 提取图像数据部分（跳过前4个字节）
                    alld = alld[1028:]  # 将处理后的数据从缓冲区中删除
                    
                    if len(imgdata) == 1024:
                        img_data = imgdata.reshape(32, 32)
                        return np.vstack((img_data[8:16, :10], img_data[7::-1, :10]))
                        # img_data = np.array(imgdata).flatten()  # 将图像数据展平成一维数组
                        
                        # # 翻转数据块，使其顺序正确
                        # for i in range(8):
                        #     start1, end1 = i * 32, (i + 1) * 32
                        #     start2, end2 = (14 - i) * 32, (15 - i) * 32
                        #     img_data[start1:end1], img_data[start2:end2] = img_data[start2:end2].copy(), img_data[start1:end1].copy()
                        
                        # img_data = np.roll(img_data, -15 * 32)  # 循环滚动数据
                        # img_data = img_data.reshape(32, 32)  # 将数据重塑为32x32的矩阵
                        # return img_data  # 返回处理后的图像矩阵

    return None  # 如果没有完整的数据包，返回None

# 在全局范围内创建 MatrixMetrics 实例
matrix_metrics = MatrixMetrics()

def update_heatmap(matrix, top_n=64):
    global latest_heatmap, heatmap_timestamp, heatmap_fig, heatmap_ax, heatmap_colorbar

    with heatmap_lock:  # 使用锁来确保热力图更新时的线程安全
        heatmap_ax.clear()  # 清除当前的热力图

        cax = heatmap_ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect=1.2)  # 显示新的矩阵热力图
        
        if heatmap_colorbar is None:
            heatmap_colorbar = heatmap_fig.colorbar(cax, ax=heatmap_ax, label='压力值')  # 如果颜色条不存在，创建一个新的
        else:
            heatmap_colorbar.update_normal(cax)  # 如果颜色条已存在，更新其数据

        # 获取矩阵的维度
        height, width = matrix.shape

        # 设置刻度标签，使其从1开始
        heatmap_ax.set_xticks(range(width))
        heatmap_ax.set_yticks(range(height))
        heatmap_ax.set_xticklabels(range(1, width + 1))
        heatmap_ax.set_yticklabels(range(1, height + 1))
        
        
        flat_indices = np.argsort(matrix.flatten())[-top_n:]  # 获取矩阵中值最大的前top_n个点的索引
        top_points = np.array(np.unravel_index(flat_indices, matrix.shape)).T  # 将一维索引转换为二维坐标

        # point_values = matrix[top_points[:, 0], top_points[:, 1]]  # 获取这些点对应的值
        # total_weight = np.sum(point_values)  # 计算这些点的总权重
        # centroid = np.sum(top_points * point_values[:, np.newaxis], axis=0) / total_weight  # 计算质心（加权平均）
            # 使用 MatrixMetrics 的方法计算重心
        
        centroid = matrix_metrics.calculate_weighted_centroid(matrix, top_n)
        # 使用PCA计算主方向
        pca = PCA(n_components=1)
        pca.fit(top_points)
        direction_vector = pca.components_[0]  # 获取主方向向量

        # 确保方向向量指向数据点的主要分布方向
        if np.dot(direction_vector, top_points.mean(axis=0) - centroid) < 0:
            direction_vector = -direction_vector  # 如果方向不正确，则翻转向量

        # 设置线段长度，并确保通过质心
        scale = max(matrix.shape) / 2
        start_point = centroid - scale * direction_vector
        end_point = centroid + scale * direction_vector

        # 获取矩阵的宽度和高度
        height, width = matrix.shape

        # 裁剪线段以确保它不会超出热力图范围
        start_point, end_point = clip_line_to_bounds(start_point, end_point, width-1, height-1)

        # 绘制主方向线
        heatmap_ax.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], 'r-', linewidth=2, label='主方向')

        # 绘制选中点和加权质心
        heatmap_ax.scatter(top_points[:, 1], top_points[:, 0], color='white', edgecolor='white', s=20, label='选中点')
        heatmap_ax.scatter(centroid[1], centroid[0], color='green', edgecolor='black', s=100, label='加权质心')

        # 显示角度信息
        angle = np.arctan2(direction_vector[0], direction_vector[1]) * 180 / np.pi  # 计算角度（从方向向量的反正切值）
        angle_text = f"角度: {angle:.2f}°"
        heatmap_ax.text(0.05, 0.95, angle_text, transform=heatmap_ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        heatmap_ax.set_title("睡姿热力图")
        heatmap_ax.set_xlabel("X轴")
        heatmap_ax.set_ylabel("Y轴")
        heatmap_ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small')

        heatmap_fig.tight_layout()  # 调整布局，使所有元素都在图内

        # 保存图像到内存
        img_buffer = io.BytesIO()
        heatmap_fig.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.1)  # 将图像保存到内存中的字节流
        img_buffer.seek(0)
        latest_heatmap = img_buffer.getvalue()  # 更新最新的热力图数据
        heatmap_timestamp = time.time()  # 记录热力图的更新时间戳

    return matrix, top_points, centroid, direction_vector, angle  # 返回矩阵、选中点、质心、方向向量和角度

def clip_line_to_bounds(start, end, width, height):
    # 裁剪线段的起点和终点，确保它们位于热力图的范围内

    def clip(p, pmin, pmax):
        # 将点p限制在pmin和pmax之间
        return max(min(p, pmax), pmin)

    def compute_intersection(p1, p2, axis, value):
        # 计算线段与指定轴上的值的交点
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

    return start, end  # 返回裁剪后的起点和终点

class DataCollectionThread(QThread):
    new_data_signal = pyqtSignal(object)  # 定义一个信号，当有新数据时发出

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ser = None  # 初始化串口连接为None

    def run(self):
        global running, matrix_buffer
        while running:
            if self.ser is None or not self.ser.is_open:
                self.ser = setup_serial_port()  # 如果串口未打开，则尝试连接
                if self.ser is None:
                    time.sleep(1)
                    continue

            try:
                frameData_b = read_matrix_from_serial(self.ser)  # 从串口读取数据
                if frameData_b is not None:
                    matrix_buffer.append(frameData_b)  # 将读取到的数据添加到矩阵缓冲区
                    self.new_data_signal.emit(frameData_b)  # 发送新数据信号
            except serial.SerialException as e:
                logging.error(f"串口读取错误: {e}")
                self.ser = None  # 如果出现错误，关闭串口并重新尝试连接
                time.sleep(1)

            if exit_event.is_set():
                break

            self.msleep(10)  # 线程休眠10毫秒

    def stop(self):
        global running
        running = False  # 设置running为False，停止线程
        self.wait()  # 等待线程安全退出
        
class InferenceThread(QThread):
    update_gui = pyqtSignal(str, float, str)  # 定义一个信号，用于更新GUI上的推理结果

    def __init__(self, inference_interval, parent=None):
        super().__init__(parent)
        self.inference_interval = inference_interval  # 推理间隔时间（秒）

    def run(self):
        global running, matrix_buffer, latest_prediction, inference_results
        last_inference_time = 0
        while running:
            current_time = time.time()
            if current_time - last_inference_time >= self.inference_interval and matrix_buffer:
                matrix = matrix_buffer[-1]  # 获取最近一次收集到的矩阵数据
                try:
                    predicted_class, confidence = predict_posture(model, matrix)  # 使用模型进行推理
                    if 0 <= predicted_class < len(posture_labels):
                        current_prediction = posture_labels[predicted_class]  # 根据预测类别索引获取对应的姿态标签
                    else:
                        current_prediction = "未知"

                    latest_prediction = {
                        "posture": current_prediction,
                        "confidence": float(confidence),
                        "timestamp": current_time
                    }

                    while not inference_results.empty():
                        inference_results.get()  # 清空之前的推理结果队列
                    inference_results.put(latest_prediction)  # 将最新推理结果放入队列
                    
                    self.update_gui.emit(current_prediction, confidence, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)))  # 发送更新信号
                    
                    last_inference_time = current_time
                except Exception as e:
                    logging.error(f"预测过程中发生错误: {e}")

            if exit_event.is_set():
                break

            self.msleep(10)  # 线程休眠10毫秒

    def stop(self):
        global running
        running = False  # 停止线程
        self.wait()  # 等待线程安全退出

class MetricsCalculationThread(QThread):
    metrics_ready = pyqtSignal(tuple, tuple, tuple, float, float, float)  # 定义一个信号，当计算完成时发出

    def __init__(self, calculation_interval, parent=None):
        super().__init__(parent)
        self.calculation_interval = calculation_interval  # 指标计算的间隔时间（秒）
        self.matrix_metrics = MatrixMetrics()  # 实例化MatrixMetrics类，用于计算矩阵的指标

    def run(self):
        global running, matrix_buffer, metrics_results
        last_calculation_time = 0
        while running:
            current_time = time.time()
            if current_time - last_calculation_time >= self.calculation_interval and matrix_buffer:
                matrix = matrix_buffer[-1]
                self.matrix_metrics.calculate(matrix)
                metrics = self.matrix_metrics.get_metrics()
                
                while not metrics_results.empty():
                    metrics_results.get()
                metrics_results.put(metrics)
                
                self.metrics_ready.emit(
                    metrics['bed_status'],
                    metrics['edge_status'],
                    metrics['centroid'],
                    metrics['rest_avg'],
                    metrics['top48_median'],
                    metrics['rest_median']
                )
                
                last_calculation_time = current_time

            if exit_event.is_set():
                break

            self.msleep(10)

    def stop(self):
        global running
        running = False  # 停止线程
        self.wait()  # 等待线程安全退出

class UIUpdateThread(QThread):
    matrix_ready = pyqtSignal(object, object, object, object, float, float)  # 定义一个信号，用于更新UI上的矩阵数据

    def __init__(self, update_interval, parent=None):
        super().__init__(parent)
        self.update_interval = update_interval  # UI更新的间隔时间（秒）

    def run(self):
        global running, matrix_buffer, latest_prediction
        last_update_time = 0
        while running:
            current_time = time.time()
            if current_time - last_update_time >= self.update_interval and matrix_buffer:
                matrix = matrix_buffer[-1]  # 获取最新的矩阵数据
                matrix, top_points, centroid, direction_vector, angle = update_heatmap(matrix)  # 更新热力图，并计算相关信息
                self.matrix_ready.emit(matrix, top_points, centroid, direction_vector, angle, current_time)  # 通过信号发送更新数据
                last_update_time = current_time

            if exit_event.is_set():
                break

            self.msleep(10)  # 线程休眠10毫秒

    def stop(self):
        global running
        running = False  # 停止线程
        self.wait()  # 等待线程安全退出

def signal_handler(sig, frame):
    print("正在退出程序...")
    global running
    running = False  # 设置running为False，以停止所有线程
    exit_event.set()  # 触发退出事件，通知所有线程停止
    QApplication.instance().quit()  # 退出PyQt5应用程序

def cleanup():
    global ser
    if ser is not None and ser.is_open:
        ser.close()  # 关闭串口连接
        print("串口连接已关闭")

def check_exit_event(app, threads):
    # 检查退出事件是否被触发，如果是则停止所有线程并退出应用
    if exit_event.is_set():
        for thread in threads:
            thread.stop()  # 停止所有正在运行的线程
        app.quit()  # 退出PyQt应用程序

def print_inference_results():
    global inference_results
    if not inference_results.empty():
        result = inference_results.get()  # 从队列中获取最新的推理结果
        # print(f"当前推理结果: {result}")  # 打印推理结果（这里被注释掉了，可能用于调试）
    else:
        print("推理结果队列为空")  # 如果队列为空，则打印提示信息

def main():
    global alld, ser, latest_prediction, running, posture_labels, matrix_buffer
    global running, matrix_buffer, latest_prediction, movement_detection_thread
    signal.signal(signal.SIGINT, signal_handler)  # 绑定SIGINT信号（如Ctrl+C）到自定义的信号处理函数
    signal.signal(signal.SIGTERM, signal_handler)  # 绑定SIGTERM信号到自定义的信号处理函数
    atexit.register(cleanup)  # 在程序退出时执行清理操作

    alld = np.array([], dtype=np.uint8)  # 初始化全局变量alld为空的uint8类型数组
    posture_labels = ['平躺', '左侧卧', '右侧卧']  # 定义睡姿标签列表

    QApplication.setFont(QFont('Arial', 10))  # 设置PyQt5应用程序的默认字体
    app = QApplication(sys.argv)  # 创建PyQt5应用程序实例
    gui = run_gui()  # 启动自定义的GUI

    # 创建并启动数据收集、推理、指标计算和UI更新的线程
    data_collection_thread = DataCollectionThread()
    inference_thread = InferenceThread(inference_interval=0.5)
    metrics_thread = MetricsCalculationThread(calculation_interval=0.5)
    ui_update_thread = UIUpdateThread(update_interval=0.5)
    # 创建和启动体动检测线程
    movement_detection_thread = MovementDetectionThread()

    threads = [data_collection_thread, inference_thread, metrics_thread, ui_update_thread, movement_detection_thread]

    for thread in threads:
        thread.start()  # 启动所有线程

    # 连接线程信号到GUI的槽
    data_collection_thread.new_data_signal.connect(gui.receive_new_data)
    inference_thread.update_gui.connect(gui.update_web_info)
    ui_update_thread.matrix_ready.connect(gui.update_heatmap)
    metrics_thread.metrics_ready.connect(gui.update_metrics)

    # 新增：连接数据收集线程到体动检测线程
    # 连接数据收集线程到体动检测线程
    data_collection_thread.new_data_signal.connect(movement_detection_thread.add_new_data)

    # 新增：连接体动检测线程到GUI的更新函数
    # movement_detection_thread.update_movement_plot.connect(gui.update_movement_plot)
    # 处理GUI信号
    gui.start_collection.connect(lambda: setattr(gui, 'collecting', True))  # 开始数据收集
    gui.pause_collection.connect(lambda: setattr(gui, 'paused', True))  # 暂停数据收集
    gui.stop_collection.connect(lambda: setattr(gui, 'collecting', False))  # 停止数据收集

    # 启动Flask
    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()  # 启动Flask服务器

    ip_addresses = get_ip_addresses()  # 获取服务器的IP地址
    print("可以通过以下地址访问web界面：")
    for ip in ip_addresses:
        print(f"http://{ip}:5000")  # 打印可访问的Web界面地址

    # 设置退出检查定时器
    exit_timer = QTimer()
    exit_timer.timeout.connect(lambda: check_exit_event(app, threads))
    exit_timer.start(100)  # 每100毫秒检查一次退出事件

    # 设置定时器来打印推理结果
    inference_print_timer = QTimer()
    inference_print_timer.timeout.connect(print_inference_results)
    inference_print_timer.start(1000)  # 每秒打印一次推理结果

    # 显示GUI
    gui.show()

    # 运行GUI事件循环
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    print("程序已成功退出")