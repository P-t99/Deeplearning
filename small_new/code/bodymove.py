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
ylimnum = 10
# 全局变量
matrix_buffer = deque(maxlen=100)  # 存储最近100帧的矩阵数据
diff_sum_buffer = np.zeros((120, 5))  # 120x5的buffer，存储差值和信息
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

def pool_matrix(matrix):
    """Pool 16x10 matrix to 8x5"""
    return matrix.reshape(8, 2, 5, 2).mean(axis=(1, 3))

def calculate_diff_sum(matrix1, matrix2):
    """Calculate frame difference sum for each column"""
    # return np.sum(np.abs(matrix2 - matrix1), axis=0)
    return np.sqrt(np.sum((matrix2 - matrix1) ** 2, axis=0))


def update_diff_sum_buffer(new_diffs):
    global diff_sum_buffer
    with data_lock:
        diff_sum_buffer = np.roll(diff_sum_buffer, -1, axis=0)
        diff_sum_buffer[-1] = new_diffs

def data_collection_thread():
    global matrix_buffer, running, diff_sum_buffer
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
                pooled_matrix = pool_matrix(matrix)
                with data_lock:
                    matrix_buffer.append(pooled_matrix)
                frame_count += 1

                if last_matrix is not None:
                    diff_sums = calculate_diff_sum(last_matrix, pooled_matrix)
                    update_diff_sum_buffer(diff_sums)

                last_matrix = pooled_matrix

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
        
def smooth_data(data, window_size=6):
    """Apply moving average smoothing to the data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def update_plot(frame):
    with data_lock:
        data = diff_sum_buffer.copy()

    # 对每一列数据进行平滑处理
    smoothed_data = np.apply_along_axis(smooth_data, 0, data)

    for i, line in enumerate(lines):
        line.set_data(range(len(smoothed_data)), smoothed_data[:, i])
        ax = axes.flatten()[i]
        ax.relim()
        ax.autoscale_view()
        ax.set_ylim(0, ylimnum)
        ax.set_title(f'Frame Diff Sum for Column {i+1}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sum of Absolute Differences')
    
    return lines

def main():
    global alld, running

    alld = np.array([], dtype=np.uint8)

    data_thread = Thread(target=data_collection_thread)
    data_thread.start()
    # 初始化Y轴范围
    for ax in axes.flatten()[:5]:
        ax.set_ylim(0, ylimnum)

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
    
