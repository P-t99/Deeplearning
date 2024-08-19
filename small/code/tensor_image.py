import serial
import serial.tools.list_ports
import numpy as np
import time
import logging
import tensorflow as tf
from flask import Flask, render_template_string, jsonify, send_file
from threading import Thread, Event
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载训练好的模型
model = tf.saved_model.load(r'D:\repository\Algo_test\ML\睡姿\Data\tensorflow\ncz\1')
logging.info("模型加载成功")

# 全局变量
latest_prediction = {"posture": "未知", "confidence": 0.0, "timestamp": time.time()}
latest_heatmap = None
heatmap_timestamp = 0
running = True
exit_event = Event()
ser = None

# Flask 应用
app = Flask(__name__)

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
            <title>睡姿预测结果</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-color: #f0f0f0; }
                #result { font-size: 24px; margin-bottom: 20px; color: #333; }
                #timestamp { font-size: 14px; color: #666; }
                #ip-addresses { font-size: 16px; margin-top: 20px; color: #333; }
                #heatmap { max-width: 100%; height: auto; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            </style>
            <script>
                function updateResult() {
                    fetch('/get_latest')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('result').innerText = `当前睡姿: ${data.posture} (置信度: ${data.confidence.toFixed(2)})`;
                            document.getElementById('timestamp').innerText = `最后更新时间: ${new Date(data.timestamp * 1000).toLocaleString()}`;
                            document.getElementById('heatmap').src = '/get_heatmap?' + data.timestamp;
                        });
                }
                setInterval(updateResult, 2000);
            </script>
        </head>
        <body>
            <h1>睡姿预测结果</h1>
            <div id="result">加载中...</div>
            <div id="timestamp"></div>
            <img id="heatmap" src="/get_heatmap" alt="热力图">
            <div id="ip-addresses">
                <p>可用的访问地址:</p>
                {% for ip in ip_addresses %}
                    <p>http://{{ip}}:5000</p>
                {% endfor %}
            </div>
        </body>
        </html>
    ''', ip_addresses=ip_addresses)

@app.route('/get_latest')
def get_latest():
    global latest_prediction, heatmap_timestamp
    return jsonify({**latest_prediction, "heatmap_timestamp": heatmap_timestamp})

@app.route('/get_heatmap')
def get_heatmap():
    global latest_heatmap
    if latest_heatmap is None:
        return "No heatmap available", 404
    
    img_io = io.BytesIO()
    latest_heatmap.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0.1)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

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
    input_data = matrix.flatten().reshape(1, 160, 1)
    input_data = tf.cast(input_data, tf.float32)
    
    predictions = model(input_data, training=False)
    predicted_class = tf.argmax(predictions[0]).numpy()
    confidence = tf.reduce_max(predictions[0]).numpy()
    return predicted_class, confidence

def find_packet_start(data):
    return np.where(np.all(np.array([data[i:i+4] for i in range(len(data)-3)]) == [170, 85, 3, 153], axis=1))[0]

def generate_heatmap(matrix, top_n=32):
    global latest_heatmap, heatmap_timestamp
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Draw the complete heatmap
    cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest', aspect=1.2)
    plt.colorbar(cax, label='压力值')

    # Get the indices of the top N values
    flat_indices = np.argsort(matrix.flatten())[-top_n:]
    top_points = np.array(np.unravel_index(flat_indices, matrix.shape)).T

    # Calculate the weighted centroid
    point_values = matrix[top_points[:, 0], top_points[:, 1]]
    total_weight = np.sum(point_values)
    centroid = np.sum(top_points * point_values[:, np.newaxis], axis=0) / total_weight

    # Perform PCA to find the main direction
    pca = PCA(n_components=1)
    pca.fit(top_points)
    direction_vector = pca.components_[0]

    # Calculate the endpoints of the direction line
    scale = max(matrix.shape) / 2
    start_point = centroid - scale * direction_vector
    end_point = centroid + scale * direction_vector

    # Draw the main direction line
    ax.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], 'r-', linewidth=2, label='主方向')

    # Highlight the selected points
    ax.scatter(top_points[:, 1], top_points[:, 0], color='white', edgecolor='black', s=50, label='选中点')

    # Highlight the centroid
    ax.scatter(centroid[1], centroid[0], color='green', edgecolor='black', s=100, label='加权质心')

    # Calculate and display the angle
    angle = np.arctan2(direction_vector[0], direction_vector[1]) * 180 / np.pi
    angle_text = f"角度: {angle:.2f}°"
    ax.text(0.05, 0.95, angle_text, transform=ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title("睡姿热力图")
    ax.set_xlabel("X轴")
    ax.set_ylabel("Y轴")
    # 将图例移动到右上角
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

    plt.tight_layout()
    
    # 更新最新的热力图
    latest_heatmap = plt.gcf()
    heatmap_timestamp = time.time()
    
    plt.close()  # 关闭图形以释放内存

def signal_handler(sig, frame):
    global running
    print("正在退出程序...")
    running = False
    exit_event.set()

def cleanup():
    global ser
    if ser is not None and ser.is_open:
        ser.close()
        print("串口连接已关闭")

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
                        img_data = imgdata.reshape(32, 32)
                        return np.vstack((img_data[8:16, :10], img_data[7::-1, :10]))
    return None

def main():
    global alld, ser, last_update_time, latest_prediction, running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)

    alld = np.array([], dtype=np.uint8)
    last_update_time = time.time()
    posture_labels = ['平躺', '左侧卧', '右侧卧']

    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    ip_addresses = get_ip_addresses()
    print("可以通过以下地址访问web界面：")
    for ip in ip_addresses:
        print(f"http://{ip}:5000")

    try:
        while running:
            if ser is None or not ser.is_open:
                ser = setup_serial_port()
                if ser is None:
                    time.sleep(1)
                    continue

            try:
                frameData_b = read_matrix_from_serial(ser)
                if frameData_b is not None:
                    current_time = time.time()
                    if current_time - last_update_time >= 2:
                        try:
                            predicted_class, confidence = predict_posture(model, frameData_b)
                            if 0 <= predicted_class < len(posture_labels):
                                current_prediction = posture_labels[predicted_class]
                            else:
                                current_prediction = "未知"
                            
                            latest_prediction = {
                                "posture": current_prediction,
                                "confidence": float(confidence),
                                "timestamp": current_time
                            }
                            
                            print(f'预测结果: {current_prediction} (置信度: {confidence:.2f})')
                            logging.info(f'预测结果: {current_prediction} (置信度: {confidence:.2f})')
                            
                            # 生成热力图
                            generate_heatmap(frameData_b)
                        except Exception as e:
                            logging.error(f"预测过程中发生错误: {e}")
                        
                        last_update_time = current_time
            except serial.SerialException as e:
                logging.error(f"串口读取错误: {e}")
                ser = None  # 重置串口连接
                time.sleep(1)

            if exit_event.is_set():
                break

            time.sleep(0.01)  # 短暂休眠以减少CPU使用

    except Exception as e:
        logging.error(f"发生未知错误: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
    print("程序已成功退出")
    
    