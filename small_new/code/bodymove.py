import serial
import serial.tools.list_ports
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from threading import Thread, Event, Lock
import logging
from queue import Queue
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
matrix_buffer = deque(maxlen=100)  # 存储最近100帧的矩阵数据
l2_norm_buffer = np.zeros((120, 5))  # 120x5的buffer，存储L2范数信息
running = True
exit_event = Event()
data_lock = Lock()

# 创建图形和轴对象
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
lines = [ax.plot([], [])[0] for ax in axes.flatten()[:5]]
plt.tight_layout()

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
                        img_data = imgdata.reshape(32, 32)
                        return np.vstack((img_data[8:16, :10], img_data[7::-1, :10]))
    return None

def normalize_matrix(matrix):
    """Normalize the matrix from 0-255 to 0-1 range"""
    return matrix.astype(float) / 255.0

def calculate_l2_norm(matrix1, matrix2):
    norms = []
    for i in range(0, 10, 2):
        col1 = np.concatenate((matrix1[:, i], matrix1[:, i+1]))
        col2 = np.concatenate((matrix2[:, i], matrix2[:, i+1]))
        norms.append(np.linalg.norm(col1 - col2))
    return norms

def update_l2_norm_buffer(new_norms):
    global l2_norm_buffer
    with data_lock:
        l2_norm_buffer = np.roll(l2_norm_buffer, -1, axis=0)
        l2_norm_buffer[-1] = new_norms

def data_collection_thread():
    global matrix_buffer, running, l2_norm_buffer
    ser = setup_serial_port()
    if ser is None:
        return

    start_time = time.time()
    frame_count = 0
    last_matrix = None

    while running:
        try:
            matrix = read_matrix_from_serial(ser)
            if matrix is not None:
                normalized_matrix = normalize_matrix(matrix)
                with data_lock:
                    matrix_buffer.append(normalized_matrix)
                frame_count += 1

                if last_matrix is not None:
                    l2_norms = calculate_l2_norm(last_matrix, normalized_matrix)
                    update_l2_norm_buffer(l2_norms)

                last_matrix = normalized_matrix

                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time >= 1:
                    frequency = frame_count / elapsed_time
                    logging.info(f"采集频率: {frequency:.2f} 帧/秒")
                    start_time = current_time
                    frame_count = 0

        except serial.SerialException as e:
            logging.error(f"串口读取错误: {e}")
            ser = setup_serial_port()
            if ser is None:
                break

        if exit_event.is_set():
            break

def update_plot(frame):
    with data_lock:
        data = l2_norm_buffer.copy()
    
    for i, line in enumerate(lines):
        line.set_data(range(len(data)), data[:, i])
        axes.flatten()[i].relim()
        axes.flatten()[i].autoscale_view()
        axes.flatten()[i].set_title(f'L2 Norm for Columns {i*2+1}-{i*2+2}')
        axes.flatten()[i].set_xlabel('Time')
        axes.flatten()[i].set_ylabel('L2 Norm')
    
    return lines

def main():
    global alld, running

    alld = np.array([], dtype=np.uint8)

    data_thread = Thread(target=data_collection_thread)
    data_thread.start()

    ani = FuncAnimation(fig, update_plot, interval=1000, blit=True)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("正在退出程序...")
    finally:
        running = False
        exit_event.set()
        data_thread.join()

if __name__ == "__main__":
    main()
    print("程序已成功退出")