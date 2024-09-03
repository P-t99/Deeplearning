import serial
import serial.tools.list_ports
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from threading import Thread, Event, Lock
import logging
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SerialMatrixReader:
    def __init__(self):
        self.alld = np.array([], dtype=np.uint8)
        self.ser = None

    def setup_serial_port(self):
        available_ports = list(serial.tools.list_ports.comports())
        for port in available_ports:
            try:
                self.ser = serial.Serial(port.device, 1000000, timeout=1)
                logging.info(f'成功连接到串口: {port.device}')
                return True
            except:
                logging.warning(f'无法连接到串口: {port.device}, 尝试下一个...')
        logging.error('未找到可用的串口设备')
        return False

    def find_packet_start(self, data):
        return np.where(np.all(np.array([data[i:i+4] for i in range(len(data)-3)]) == [170, 85, 3, 153], axis=1))[0]

    def read_matrix_from_serial(self):
        if self.ser.in_waiting > 0:
            receive = self.ser.read(self.ser.in_waiting)
            self.alld = np.concatenate([self.alld, np.frombuffer(receive, dtype=np.uint8)])

            if len(self.alld) >= 4:
                index = self.find_packet_start(self.alld)
                if len(index) > 0:
                    if index[0] > 0:
                        self.alld = self.alld[index[0]:]

                    if len(self.alld) >= 1028:
                        imgdata = self.alld[4:1028]
                        self.alld = self.alld[1028:]

                        if len(imgdata) == 1024:
                            img_data = imgdata.reshape(32, 32)
                            return np.vstack((img_data[8:16, :10], img_data[7::-1, :10]))
        return None


class DataCollector:
    def __init__(self, ylimnum=10):
        self.matrix_buffer = deque(maxlen=100)  # 存储最近100帧的矩阵数据
        self.diff_sum_buffer = np.zeros((120, 5))  # 120x5的buffer，存储差值和信息
        self.running = True
        self.exit_event = Event()
        self.data_lock = Lock()
        self.serial_reader = SerialMatrixReader()
        self.ylimnum = ylimnum

    def pool_matrix(self, matrix):
        """Pool 16x10 matrix to 8x5"""
        return matrix.reshape(8, 2, 5, 2).mean(axis=(1, 3))

    def calculate_diff_sum(self, matrix1, matrix2):
        """Calculate frame difference sum for each column"""
        return np.sqrt(np.sum((matrix2 - matrix1) ** 2, axis=0))

    def update_diff_sum_buffer(self, new_diffs):
        with self.data_lock:
            self.diff_sum_buffer = np.roll(self.diff_sum_buffer, -1, axis=0)
            self.diff_sum_buffer[-1] = new_diffs

    def data_collection_thread(self):
        if not self.serial_reader.setup_serial_port():
            return

        start_time = time.time()
        frame_count = 0
        last_matrix = None

        while self.running:
            try:
                matrix = self.serial_reader.read_matrix_from_serial()
                if matrix is not None:
                    pooled_matrix = self.pool_matrix(matrix)
                    with self.data_lock:
                        self.matrix_buffer.append(pooled_matrix)
                    frame_count += 1

                    if last_matrix is not None:
                        diff_sums = self.calculate_diff_sum(last_matrix, pooled_matrix)
                        self.update_diff_sum_buffer(diff_sums)

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
                if not self.serial_reader.setup_serial_port():
                    break

            if self.exit_event.is_set():
                break

    def smooth_data(self, data, window_size=6):
        """Apply moving average smoothing to the data."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')


class DataVisualizer:
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.fig, self.axes = plt.subplots(5, 1, figsize=(10, 15))
        self.lines = [ax.plot([], [])[0] for ax in self.axes]
        plt.tight_layout()

        # 初始化Y轴范围
        for ax in self.axes:
            ax.set_ylim(0, self.data_collector.ylimnum)

    def update_plot(self, frame):
        with self.data_collector.data_lock:
            data = self.data_collector.diff_sum_buffer.copy()

        # 对每一列数据进行平滑处理
        smoothed_data = np.apply_along_axis(self.data_collector.smooth_data, 0, data)

        for i, line in enumerate(self.lines):
            line.set_data(range(len(smoothed_data)), smoothed_data[:, i])
            ax = self.axes[i]
            ax.relim()
            ax.autoscale_view()
            ax.set_ylim(0, self.data_collector.ylimnum)

                # 使用 ax.text 将标题放置在图表下方
            # ax.text(0.5, -0.2, f'Column {i+1}', fontsize=10, ha='center', transform=ax.transAxes)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'Column {i+1}')

        return self.lines

    def start_visualization(self):
        ani = FuncAnimation(self.fig, self.update_plot, interval=1000, blit=True)
        try:
            plt.show()
        except KeyboardInterrupt:
            print("正在退出程序...")


def main():
    data_collector = DataCollector(ylimnum=10)

    data_thread = Thread(target=data_collector.data_collection_thread)
    data_thread.start()

    visualizer = DataVisualizer(data_collector)
    visualizer.start_visualization()

    data_collector.running = False
    data_collector.exit_event.set()
    data_thread.join()


if __name__ == "__main__":
    main()
    print("程序已成功退出")
