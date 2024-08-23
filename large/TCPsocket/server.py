import serial
import serial.tools.list_ports
import serial_asyncio
import numpy as np
import time
import socket
import struct
import logging
import signal
import json
import asyncio
from aiohttp import web
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
serial_connected = False
clients = set()
config = {
    "port": 64990,
    "baud_rate": 1000000,
    "matrix_size": 32
}

# 使用 deque 来存储最新的 100 帧矩阵数据
matrix_buffer = deque(maxlen=100)
send_queue = asyncio.Queue(maxsize=100)  # 用于存储待发送的帧
exit_event = asyncio.Event()

# 计数器
extracted_matrix_count = 0  # 从串口中提取的矩阵个数
processed_matrix_count = 0  # 处理后的矩阵个数

class FrameData:
    def __init__(self, matrix, timestamp):
        self.matrix = matrix
        self.timestamp = timestamp
        self.frame_id = int(timestamp * 1000)  # 使用毫秒级时间戳作为帧ID

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logging.error(f"获取本地 IP 地址时出错: {e}")
        return "127.0.0.1"

def load_config():
    global config
    try:
        with open('config.json', 'r') as f:
            config.update(json.load(f))
    except FileNotFoundError:
        logging.warning("配置文件未找到，使用默认配置")
        save_config()

def save_config():
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

async def setup_serial_port(loop):
    global serial_connected
    while not exit_event.is_set():
        available_ports = list(serial.tools.list_ports.comports())
        for port in available_ports:
            try:
                ser_protocol, _ = await serial_asyncio.create_serial_connection(
                    loop,
                    lambda: SerialProtocol(),
                    port.device,
                    baudrate=config['baud_rate']
                )
                logging.info(f'成功连接到串口: {port.device}')
                serial_connected = True
                return ser_protocol
            except:
                logging.warning(f'无法连接到串口: {port.device}, 尝试下一个...')
        logging.error('未找到可用的串口设备，30秒后重试...')
        try:
            await asyncio.wait_for(exit_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass
    return None

class SerialProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        self.transport = transport
        logging.info('Serial port opened')

    def data_received(self, data):
        asyncio.create_task(self.process_data(data))

    async def process_data(self, data):
        global extracted_matrix_count
        buffer = bytearray(data)
        while len(buffer) >= 1028:
            if buffer[:4] == b'\xaa\x55\x03\x99':
                imgdata = buffer[4:1028]
                buffer = buffer[1028:]
                
                img_data = np.frombuffer(imgdata, dtype=np.uint8).reshape(32, 32)
                
                # Swap and roll matrix data
                for i in range(8):
                    img_data[i], img_data[14-i] = img_data[14-i].copy(), img_data[i].copy()
                img_data = np.roll(img_data, -15, axis=0)
                
                extracted_matrix_count += 1
                
                frame = FrameData(img_data, time.time())
                matrix_buffer.append(frame)
                await send_queue.put(frame)
            else:
                buffer = buffer[1:]

async def send_frame_data(writer, frame):
    global processed_matrix_count
    frame_header = struct.pack('!I', 0xAA55AAAA)
    data_length = struct.pack('!I', config['matrix_size'] * config['matrix_size'])
    timestamp = struct.pack('!I', int(frame.timestamp))
    frame_id = struct.pack('!Q', frame.frame_id)
    matrix_bytes = frame.matrix.tobytes()
    checksum = struct.pack('!H', sum(matrix_bytes) & 0xFFFF)
    serial_status = struct.pack('!?', serial_connected)
    
    data_to_send = frame_header + data_length + timestamp + frame_id + matrix_bytes + checksum + serial_status
    
    writer.write(data_to_send)
    await writer.drain()
    processed_matrix_count += 1

async def send_heartbeat(writer):
    heartbeat_data = struct.pack('!I', 0xBEADBEEF)  # 使用有效的十六进制标识符
    writer.write(heartbeat_data)
    await writer.drain()

async def handle_client(reader, writer):
    global clients
    addr = writer.get_extra_info('peername')
    clients.add(writer)
    logging.info(f"New client connected: {addr}")
    
    last_sent_frame_id = 0
    last_activity_time = time.time()
    
    try:
        while not exit_event.is_set():
            try:
                frame = await asyncio.wait_for(send_queue.get(), timeout=5.0)
                if frame.frame_id > last_sent_frame_id:
                    await send_frame_data(writer, frame)
                    last_sent_frame_id = frame.frame_id
                    last_activity_time = time.time()
                send_queue.task_done()
            except asyncio.TimeoutError:
                # 如果5秒内没有新帧，发送心跳包
                if time.time() - last_activity_time > 5:
                    await send_heartbeat(writer)
                    last_activity_time = time.time()
    except ConnectionResetError:
        logging.info(f"Client disconnected: {addr}")
    finally:
        clients.remove(writer)
        writer.close()

async def serial_reader(loop):
    global serial_connected
    ser_protocol = await setup_serial_port(loop)
    if ser_protocol is None:
        return
    
    try:
        while not exit_event.is_set():
            await asyncio.sleep(1)  # 保持任务活跃
    except Exception as e:
        logging.error(f"Error in serial reader: {e}")
    finally:
        if hasattr(ser_protocol, 'transport'):
            ser_protocol.transport.close()

async def log_status():
    while not exit_event.is_set():
        logging.info(f"Extracted matrices: {extracted_matrix_count}")
        logging.info(f"Processed matrices: {processed_matrix_count}")
        logging.info("-" * 50)
        await asyncio.sleep(1)

async def health_check(request):
    return web.Response(text="OK")

async def shutdown(app):
    exit_event.set()
    for client in clients:
        client.close()
    await asyncio.sleep(1)  # 给一些时间让客户端连接关闭

async def main():
    load_config()
    
    loop = asyncio.get_running_loop()
    
    # 创建并配置aiohttp应用
    app = web.Application()
    app.add_routes([web.get('/health', health_check)])
    app.on_shutdown.append(shutdown)
    
    # 启动Web服务器（用于健康检查）
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    
    # 启动串口读取任务
    serial_task = asyncio.create_task(serial_reader(loop))
    
    # 启动状态记录任务
    log_task = asyncio.create_task(log_status())
    
    # 启动主服务器
    local_ip = get_local_ip()
    server = await asyncio.start_server(handle_client, '0.0.0.0', config['port'])
    logging.info(f'Server running on {local_ip}:{config["port"]}')
    print(f"Please connect to the client using the following address: {local_ip}:{config['port']}")
    
    async with server:
        await server.serve_forever()
    
    # 等待所有任务完成
    await asyncio.gather(serial_task, log_task)
    await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
    finally:
        exit_event.set()
        logging.info("Server has been shut down")