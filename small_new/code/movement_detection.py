import numpy as np
import time
import logging
from collections import deque
from threading import Thread, Event, Lock
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class DataCollector:
    def __init__(self, ylimnum=10):
        self.buffer = deque(maxlen=100)  # 存储最近100帧的矩阵数据
        self.diff_sum_buffer = np.zeros((120, 5))  # 120x5的buffer，存储差值和信息
        self.running = True
        self.exit_event = Event()
        self.data_lock = Lock()
        self.ylimnum = ylimnum
        self.matrix = None

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

    def process_matrix(self, matrix):
        if matrix is not None:
            pooled_matrix = self.pool_matrix(matrix)
            with self.data_lock:
                self.buffer.append(pooled_matrix)

            if len(self.buffer) > 1:
                last_matrix = self.buffer[-2]
                diff_sums = self.calculate_diff_sum(last_matrix, pooled_matrix)
                self.update_diff_sum_buffer(diff_sums)

    def smooth_data(self, data, window_size=6):
        """Apply moving average smoothing to the data."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

class DataVisualizer:
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.fig = Figure(figsize=(10, 15))
        self.axes = self.fig.subplots(5, 1)
        self.lines = [ax.plot([], [])[0] for ax in self.axes]
        self.fig.tight_layout()

        for ax in self.axes:
            ax.set_ylim(0, self.data_collector.ylimnum)

    def update_plot(self):
        with self.data_collector.data_lock:
            data = self.data_collector.diff_sum_buffer.copy()

        smoothed_data = np.apply_along_axis(self.data_collector.smooth_data, 0, data)

        for i, line in enumerate(self.lines):
            line.set_data(range(len(smoothed_data)), smoothed_data[:, i])
            ax = self.axes[i]
            ax.relim()
            ax.autoscale_view()
            ax.set_ylim(0, self.data_collector.ylimnum)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'Column {i+1}')

        return self.fig

    def get_plot_image(self):
        self.update_plot()
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        return img