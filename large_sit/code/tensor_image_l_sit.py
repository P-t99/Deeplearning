import serial
import serial.tools.list_ports
import numpy as np
import time
import logging
import tensorflow as tf
from flask import Flask, render_template_string, jsonify, send_file
from threading import Thread, Event, Lock
import socket
import netifaces
import signal
import sys
import atexit
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import io
from queue import Queue
from collections import deque

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from gui_module import run_gui  # 确保gui_module.py在同一目录下

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
global matrix_buffer, latest_prediction, latest_heatmap, heatmap_timestamp, running, exit_event, ser, alld, posture_labels
matrix_buffer = deque(maxlen=100)  # 存储最近100帧矩阵数据
latest_prediction = {"posture": "未知", "confidence": 0.0, "timestamp": time.time()}
latest_heatmap = None
heatmap_timestamp = 0
heatmap_lock = Lock()
running = True
exit_event = Event()
ser = None
alld = np.array([], dtype=np.uint8)
posture_labels = ['平躺', '左侧卧', '右侧卧','坐起']

# 创建热力图figure和axes（在主线程中）
heatmap_fig, heatmap_ax = plt.subplots(figsize=(10, 8))
heatmap_colorbar = None

inference_results = Queue(maxsize=1)  # 存储最新的推理结果
metrics_results = Queue(maxsize=1)  # 存储最新的指标计算结果

# 加载训练好的模型
model = tf.saved_model.load(r'D:\repository\deeplearning\large_sit\Data\tensorflow\ncz\1')
logging.info("模型加载成功")

# Flask 应用
app = Flask(__name__)

class MatrixMetrics:
    def __init__(self):
        self.top48_avg = 0
        self.rest_avg = 0
        self.difference_percentage = 0
        self.top48_median = 0
        self.rest_median = 0
        self.difference_percentage_median = 0

    def calculate(self, matrix):
        flat_matrix = matrix.flatten()
        sorted_matrix = np.sort(flat_matrix)[::-1]  # 降序排序
        
        # 平均值计算
        self.top48_avg = np.mean(sorted_matrix[:32]) if len(sorted_matrix) >= 32 else 0
        self.rest_avg = np.mean(sorted_matrix[32:]) if len(sorted_matrix) > 32 else 0
        self.difference_percentage = ((self.top48_avg - self.rest_avg) / 255 * 100) if self.top48_avg != self.rest_avg else 0
        
        # 中位数计算
        self.top48_median = (np.median(sorted_matrix[1:64]) + np.median(sorted_matrix[1:128])) / 2 if len(sorted_matrix) >= 128 else 0
        rest_elements_after_32 = sorted_matrix[300:]
        non_zero_elements = rest_elements_after_32[rest_elements_after_32 > 5]
        self.rest_median = np.mean(non_zero_elements) if len(non_zero_elements) > 0 else 5
        self.difference_percentage_median = ((self.top48_median - self.rest_median) / 255 * 100) if self.top48_median != self.rest_median else 0
    
    def get_metrics(self):
        return {
            "top48_avg": self.top48_avg,
            "rest_avg": self.rest_avg,
            "difference_percentage": self.difference_percentage,
            "top48_median": self.top48_median,
            "rest_median": self.rest_median,
            "difference_percentage_median": self.difference_percentage_median
        }


def get_ip_addresses():
    ip_addresses = []
    interfaces = netifaces.interfaces()
    for interface in interfaces:
        if interface == 'lo':
            continue
        iface = netifaces.ifaddresses(interface).get(netifaces.AF_INET)
        if iface:
            for addr in iface:
                ip_addresses.append(addr['addr'])
    return ip_addresses

@app.route('/')
def home():
    ip_addresses = get_ip_addresses()
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
                                <p>Top48均值: ${data.top48_avg.toFixed(2)}</p>
                                <p>其余均值: ${data.rest_avg.toFixed(2)}</p>
                                <p>差值百分比: ${data.difference_percentage.toFixed(2)}%</p>
                                <p>Top48中位数: ${data.top48_median.toFixed(2)}</p>
                                <p>其余中位数: ${data.rest_median.toFixed(2)}</p>
                                <p>中位数差值百分比: ${data.difference_percentage_median.toFixed(2)}%</p>
                            `;
                            document.getElementById('heatmap').src = '/get_heatmap?' + new Date().getTime();
                        });
                }
                setInterval(updateResult, 1500);
            </script>
        </head>
        <body>
            <div class="container">
                <h1>睡姿预测结果</h1>
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
                <div id="ip-addresses">
                    <p>可用的访问地址:</p>
                    {% for ip in ip_addresses %}
                        <p>http://{{ip}}:5000</p>
                    {% endfor %}
                </div>
            </div>
        </body>
        </html>
    ''', ip_addresses=ip_addresses)


@app.route('/get_latest')
def get_latest():
    global latest_prediction, heatmap_timestamp
    try:
        current_prediction = inference_results.get(block=False)
    except Queue.Empty:
        current_prediction = latest_prediction
    
    try:
        metrics = metrics_results.get(block=False)
    except Queue.Empty:
        metrics = {}
    
    # print(f"Web请求返回的预测结果: {current_prediction}")  # 调试输出
    return jsonify({**current_prediction, "heatmap_timestamp": heatmap_timestamp, **metrics})

@app.route('/get_heatmap')
def get_heatmap():
    global latest_heatmap
    with heatmap_lock:
        if latest_heatmap is None:
            return "No heatmap available", 404
        return send_file(io.BytesIO(latest_heatmap), mimetype='image/png')

def run_flask_app():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def setup_serial_port():
    available_ports = list(serial.tools.list_ports.comports())
    for port in available_ports:
        try:
            s = serial.Serial(port.device, 1000000, timeout=1)
            logging.info(f'成功连接到串口: {port.device}')
            return s
        except:
            logging.warning(f'无法连接到串口: {port.device}, 尝试下一个...')
    logging.error('未找到可用的串口设备')
    return None

def predict_posture(model, matrix):
    input_data = matrix.flatten().reshape(1, 1024, 1)
    input_data = tf.cast(input_data, tf.float32)
    
    predictions = model(input_data, training=False)
    predicted_class = tf.argmax(predictions[0]).numpy()
    confidence = tf.reduce_max(predictions[0]).numpy()
    return predicted_class, confidence

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

def update_heatmap(matrix, top_n=400):
    global latest_heatmap, heatmap_timestamp, heatmap_fig, heatmap_ax, heatmap_colorbar

    with heatmap_lock:
        heatmap_ax.clear()

        cax = heatmap_ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect=1.5)
        
        if heatmap_colorbar is None:
            heatmap_colorbar = heatmap_fig.colorbar(cax, ax=heatmap_ax, label='压力值')
        else:
            heatmap_colorbar.update_normal(cax)

        flat_indices = np.argsort(matrix.flatten())[-top_n:]
        top_points = np.array(np.unravel_index(flat_indices, matrix.shape)).T

        point_values = matrix[top_points[:, 0], top_points[:, 1]]
        total_weight = np.sum(point_values)
        centroid = np.sum(top_points * point_values[:, np.newaxis], axis=0) / total_weight

        # 使用PCA计算主方向
        pca = PCA(n_components=1)
        pca.fit(top_points)
        direction_vector = pca.components_[0]

        # 确保方向向量指向数据点的主要分布方向
        if np.dot(direction_vector, top_points.mean(axis=0) - centroid) < 0:
            direction_vector = -direction_vector

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
        angle = np.arctan2(direction_vector[0], direction_vector[1]) * 180 / np.pi
        angle_text = f"角度: {angle:.2f}°"
        heatmap_ax.text(0.05, 0.95, angle_text, transform=heatmap_ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        heatmap_ax.set_title("睡姿热力图")
        heatmap_ax.set_xlabel("X轴")
        heatmap_ax.set_ylabel("Y轴")
        heatmap_ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small')

        heatmap_fig.tight_layout()

        # 保存图像到内存
        img_buffer = io.BytesIO()
        heatmap_fig.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.1)
        img_buffer.seek(0)
        latest_heatmap = img_buffer.getvalue()
        heatmap_timestamp = time.time()

    return matrix, top_points, centroid, direction_vector, angle

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




class DataCollectionThread(QThread):
    new_data_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ser = None

    def run(self):
        global running, matrix_buffer
        while running:
            if self.ser is None or not self.ser.is_open:
                self.ser = setup_serial_port()
                if self.ser is None:
                    time.sleep(1)
                    continue

            try:
                frameData_b = read_matrix_from_serial(self.ser)
                if frameData_b is not None:
                    # print(f"New matrix data collected: shape {frameData_b.shape}")
                    matrix_buffer.append(frameData_b)
                    self.new_data_signal.emit(frameData_b)
            except serial.SerialException as e:
                logging.error(f"串口读取错误: {e}")
                self.ser = None
                time.sleep(1)

            if exit_event.is_set():
                break

            self.msleep(10)

    def stop(self):
        global running
        running = False
        self.wait()

class InferenceThread(QThread):
    update_gui = pyqtSignal(str, float, str)

    def __init__(self, inference_interval, parent=None):
        super().__init__(parent)
        self.inference_interval = inference_interval

    def run(self):
        global running, matrix_buffer, latest_prediction, inference_results
        last_inference_time = 0
        while running:
            current_time = time.time()
            if current_time - last_inference_time >= self.inference_interval and matrix_buffer:
                matrix = matrix_buffer[-1]
                # print(f"Running inference. Matrix buffer size: {len(matrix_buffer)}")
                try:
                    predicted_class, confidence = predict_posture(model, matrix)
                    if 0 <= predicted_class < len(posture_labels):
                        current_prediction = posture_labels[predicted_class]
                    else:
                        current_prediction = "未知"
                    
                    latest_prediction = {
                        "posture": current_prediction,
                        "confidence": float(confidence),
                        "timestamp": current_time
                    }
                    
                    # print(f"新预测结果: {latest_prediction}")
                    
                    while not inference_results.empty():
                        inference_results.get()
                    inference_results.put(latest_prediction)
                    
                    self.update_gui.emit(current_prediction, confidence, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)))
                    
                    last_inference_time = current_time
                except Exception as e:
                    logging.error(f"预测过程中发生错误: {e}")
                    print(f"预测错误: {e}")

            if exit_event.is_set():
                break

            self.msleep(10)

    def stop(self):
        global running
        running = False
        self.wait()

class MetricsCalculationThread(QThread):
    metrics_ready = pyqtSignal(float, float, float, float, float, float)

    def __init__(self, calculation_interval, parent=None):
        super().__init__(parent)
        self.calculation_interval = calculation_interval
        self.matrix_metrics = MatrixMetrics()

    def run(self):
        global running, matrix_buffer, metrics_results
        last_calculation_time = 0
        while running:
            current_time = time.time()
            if current_time - last_calculation_time >= self.calculation_interval and matrix_buffer:
                matrix = matrix_buffer[-1]
                # print(f"Calculating metrics. Matrix buffer size: {len(matrix_buffer)}")
                self.matrix_metrics.calculate(matrix)
                metrics = self.matrix_metrics.get_metrics()
                
                while not metrics_results.empty():
                    metrics_results.get()
                metrics_results.put(metrics)
                
                self.metrics_ready.emit(
                    metrics['top48_avg'],
                    metrics['rest_avg'],
                    metrics['difference_percentage'],
                    metrics['top48_median'],
                    metrics['rest_median'],
                    metrics['difference_percentage_median']
                )
                
                last_calculation_time = current_time

            if exit_event.is_set():
                break

            self.msleep(10)

    def stop(self):
        global running
        running = False
        self.wait()

class UIUpdateThread(QThread):
    matrix_ready = pyqtSignal(object, object, object, object, float, float)

    def __init__(self, update_interval, parent=None):
        super().__init__(parent)
        self.update_interval = update_interval

    def run(self):
        global running, matrix_buffer, latest_prediction
        last_update_time = 0
        while running:
            current_time = time.time()
            if current_time - last_update_time >= self.update_interval and matrix_buffer:
                matrix = matrix_buffer[-1]
                # print(f"Updating UI. Matrix buffer size: {len(matrix_buffer)}")
                matrix, top_points, centroid, direction_vector, angle = update_heatmap(matrix)
                self.matrix_ready.emit(matrix, top_points, centroid, direction_vector, angle, current_time)
                last_update_time = current_time

            if exit_event.is_set():
                break

            self.msleep(10)

    def stop(self):
        global running
        running = False
        self.wait()

def signal_handler(sig, frame):
    print("正在退出程序...")
    global running
    running = False
    exit_event.set()
    QApplication.instance().quit()

def cleanup():
    global ser
    if ser is not None and ser.is_open:
        ser.close()
        print("串口连接已关闭")

def check_exit_event(app, threads):
    if exit_event.is_set():
        for thread in threads:
            thread.stop()
        app.quit()

def print_inference_results():
    global inference_results
    if not inference_results.empty():
        result = inference_results.get()
        # print(f"当前推理结果: {result}")
    else:
        print("推理结果队列为空")

def main():
    global alld, ser, latest_prediction, running, posture_labels, matrix_buffer

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)

    alld = np.array([], dtype=np.uint8)
    posture_labels = ['平躺', '左侧卧', '右侧卧','坐起']

    QApplication.setFont(QFont('Arial', 10))
    app = QApplication(sys.argv)
    gui = run_gui()
    
    # 创建并启动线程
    data_collection_thread = DataCollectionThread()
    inference_thread = InferenceThread(inference_interval=0.5)
    metrics_thread = MetricsCalculationThread(calculation_interval=0.5)
    ui_update_thread = UIUpdateThread(update_interval=0.5)

    threads = [data_collection_thread, inference_thread, metrics_thread, ui_update_thread]

    for thread in threads:
        thread.start()

    # 连接信号
    data_collection_thread.new_data_signal.connect(gui.receive_new_data)
    inference_thread.update_gui.connect(gui.update_web_info)
    ui_update_thread.matrix_ready.connect(gui.update_heatmap)
    metrics_thread.metrics_ready.connect(gui.update_metrics)

    gui.start_collection.connect(lambda: setattr(gui, 'collecting', True))
    gui.pause_collection.connect(lambda: setattr(gui, 'paused', True))
    gui.stop_collection.connect(lambda: setattr(gui, 'collecting', False))

    # 启动Flask
    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    ip_addresses = get_ip_addresses()
    print("可以通过以下地址访问web界面：")
    for ip in ip_addresses:
        print(f"http://{ip}:5000")

    # 设置退出检查定时器
    exit_timer = QTimer()
    exit_timer.timeout.connect(lambda: check_exit_event(app, threads))
    exit_timer.start(100)

    # 设置定时器来打印推理结果
    inference_print_timer = QTimer()
    inference_print_timer.timeout.connect(print_inference_results)
    inference_print_timer.start(1000)  # 每秒打印一次

    # 显示GUI
    gui.show()

    # 运行GUI事件循环
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    print("程序已成功退出")