import socket
import time
import asyncio
from bleak import BleakScanner
from datetime import datetime
import threading

# Create a socket instance
clientsocket = socket.socket()

# Connect to the server (assuming it's hosted on localhost and port 8000)
server_ip = 'localhost'  # Change this to the actual IP if the server is remote
port = 8000

def socket_client():
    try:
        # Attempt to connect to the server
        clientsocket.connect((server_ip, port))
        print(f"Connected to server at {server_ip} on port {port}")
        
        while True:
            time.sleep(5)  # Keep the connection open and sleep
    except Exception as e:
        print(f"An error occurred in socket client: {e}")
    finally:
        # Clean up the connection
        clientsocket.close()
        print("Socket connection closed")

async def scan_bluetooth_devices():
    print("Scanning for bluetooth devices...")
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        devices = await BleakScanner.discover()
        
        print(f"\nScan completed at {scan_time}")
        print(f"Found {len(devices)} devices:\n")
        
        for device in devices:
            device_info = f"Device Name: {device.name or 'Unknown'}, MAC Address: {device.address}"
            print(device_info)
            # Send the device info to the socket server
            send_to_server(device_info)
            if device.address == 'B0:E4:5C:37:97:1F':
                send_to_server(f"Device Name: DHOM's Phone, MAC Address: {device.address}")
            
    except Exception as e:
        print(f"An error occurred in Bluetooth scan: {str(e)}")

def send_to_server(message):
    try:
        clientsocket.send(message.encode('utf-8'))
        print(f"Sent to server: {message}")
    except Exception as e:
        print(f"Error sending to server: {e}")

async def main():
    print("Starting Bluetooth scan...")
    await scan_bluetooth_devices()

if __name__ == "__main__":
    # Start the socket client in a separate thread
    socket_thread = threading.Thread(target=socket_client)
    socket_thread.start()

    # Run the async function for scanning Bluetooth devices
    asyncio.run(main())
