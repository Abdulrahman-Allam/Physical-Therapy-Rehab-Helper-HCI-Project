import socket
import time
import asyncio
from bleak import BleakScanner
from datetime import datetime
import threading

# Define server IP and port
host_ip = '0.0.0.0'  # Listen on all available interfaces
port = 8000

# Initialize the server socket and set it up to accept connections
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host_ip, port))
server_socket.listen(1)  # Listen for 1 client connection at a time

client_socket = None
is_client_connected = False

def socket_server():
    global client_socket, is_client_connected
    print(f"Server listening on {host_ip}:{port}...")

    try:
        # Wait for a client to connect
        client_socket, client_address = server_socket.accept()
        print(f"Connected to client at {client_address}")
        is_client_connected = True  # Mark client as connected

        while is_client_connected:
            time.sleep(5)  # Keep the connection open and sleep
    except Exception as e:
        print(f"An error occurred in socket server: {e}")
    finally:
        # Clean up the connection
        if client_socket:
            client_socket.close()
        server_socket.close()
        is_client_connected = False
        print("Socket server closed")

async def scan_bluetooth_devices():
    # Wait until a client is connected before scanning
    while not is_client_connected:
        print("Waiting for client connection to start Bluetooth scan...")
        await asyncio.sleep(1)

    print("Scanning for Bluetooth devices...")
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        devices = await BleakScanner.discover()
        
        print(f"\nScan completed at {scan_time}")
        print(f"Found {len(devices)} devices:\n")
        
        for device in devices:
            device_info = f"Device Name: {device.name or 'Unknown'}, MAC Address: {device.address}"
            print(device_info)
            # Send the device info to the client
            send_to_client(device_info)
            if device.address == 'B0:E4:5C:37:97:1F':
                send_to_client(f"Device Name: DHOM's Phone, MAC Address: {device.address}")
            
    except Exception as e:
        print(f"An error occurred in Bluetooth scan: {str(e)}")

def send_to_client(message):
    try:
        if is_client_connected and client_socket:
            client_socket.send(message.encode('utf-8'))
            print(f"Sent to client: {message}")
        else:
            print("Client not connected, cannot send data.")
    except Exception as e:
        print(f"Error sending to client: {e}")

async def main():
    print("Starting Bluetooth scan (after client connection)...")
    await scan_bluetooth_devices()

if __name__ == "__main__":
    # Start the socket server in a separate thread
    server_thread = threading.Thread(target=socket_server)
    server_thread.start()

    # Run the async function for scanning Bluetooth devices
    asyncio.run(main())
