import serial
import serial.tools.list_ports
import numpy as np
import time

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

def read_matrix_from_serial(ser):
    alld = np.array([], dtype=np.uint8)
    while True:
        if ser.in_waiting > 0:
            receive = ser.read(ser.in_waiting)
            alld = np.concatenate([alld, np.frombuffer(receive, dtype=np.uint8)])
            
            if len(alld) >= 1028:
                print(f"接收到 {len(alld)} 字节数据")
                if np.all(alld[:4] == [170, 85, 3, 153]):
                    imgdata = alld[4:1028]
                    alld = alld[1028:]
                    
                    if len(imgdata) == 1024:
                        img_data = np.array(imgdata).flatten()
                        
                        for i in range(8):
                            start1, end1 = i * 32, (i + 1) * 32
                            start2, end2 = (14 - i) * 32, (15 - i) * 32
                            img_data[start1:end1], img_data[start2:end2] = img_data[start2:end2].copy(), img_data[start1:end1].copy()
                        
                        img_data = np.roll(img_data, -15 * 32)
                        return img_data.reshape(32, 32)
                else:
                    alld = alld[1:]
        time.sleep(0.01)

def main():
    ser = setup_serial_port()
    if ser is None:
        print("无法设置串口，退出程序")
        return

    print("开始读取串口数据...")
    try:
        while True:
            matrix = read_matrix_from_serial(ser)
            if matrix is not None:
                print("\n读取到新的矩阵数据:")
                print(f"矩阵形状: {matrix.shape}")
                print(f"矩阵数据 (前10个元素): {matrix.flatten()[:10]}")
                print(f"矩阵最小值: {np.min(matrix)}, 最大值: {np.max(matrix)}")
                print(f"矩阵平均值: {np.mean(matrix):.2f}")
                print(f"非零元素数量: {np.count_nonzero(matrix)}")
                print("-" * 50)
    except KeyboardInterrupt:
        print("\n接收到退出信号，正在关闭串口...")
    finally:
        ser.close()
        print("串口已关闭")

if __name__ == "__main__":
    main()