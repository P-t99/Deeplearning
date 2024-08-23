import asyncio
import struct
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MatrixClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        self.running = False

    async def connect(self):
        try:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            logging.info(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
            return False

    async def receive_data(self):
        try:
            # Read frame header
            frame_header = await self.reader.readexactly(4)
            if frame_header == struct.pack('!I', 0xBEADBEEF):
                logging.info("Received heartbeat")
                return None

            # Read data length
            data_length = struct.unpack('!I', await self.reader.readexactly(4))[0]

            # Read timestamp
            timestamp = struct.unpack('!I', await self.reader.readexactly(4))[0]

            # Read frame ID
            frame_id = struct.unpack('!Q', await self.reader.readexactly(8))[0]

            # Read matrix data
            matrix_data = await self.reader.readexactly(data_length)
            matrix = np.frombuffer(matrix_data, dtype=np.uint8).reshape(32, 32)

            # Read checksum
            checksum = struct.unpack('!H', await self.reader.readexactly(2))[0]

            # Read serial status
            serial_status = struct.unpack('!?', await self.reader.readexactly(1))[0]

            logging.info(f"Received matrix: Frame ID {frame_id}, Timestamp {timestamp}")
            return matrix, timestamp, frame_id, serial_status

        except asyncio.IncompleteReadError:
            logging.error("Connection closed by server")
            self.running = False
            return None
        except Exception as e:
            logging.error(f"Error receiving data: {e}")
            return None

    async def run(self):
        self.running = True
        while self.running:
            data = await self.receive_data()
            if data:
                matrix, timestamp, frame_id, serial_status = data
                # Here you can process or display the matrix as needed
                logging.info(f"Matrix sum: {np.sum(matrix)}, Serial status: {serial_status}")
            await asyncio.sleep(0.01)  # Small delay to prevent busy-waiting

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.running = False
        logging.info("Client closed")

async def main():
    client = MatrixClient('localhost', 64990)  # Replace with your server's IP and port
    if await client.connect():
        try:
            await client.run()
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, shutting down...")
        finally:
            await client.close()

if __name__ == "__main__":
    asyncio.run(main())