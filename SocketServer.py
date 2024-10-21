# Imports modules
import socket
import time

# Create a socket instance
listensocket = socket.socket() 
Port = 8000  # Port to host server on
maxConnections = 999
IP = socket.gethostname()  # IP address of local machine

# Bind the socket
listensocket.bind(('localhost', Port))

# Starts server
listensocket.listen(maxConnections)
print(f"Server started at {IP} on port {Port}")

# Accept the incoming connection
(clientsocket, address) = listensocket.accept()
print("New connection made!")

running = True

while running:
    try:
        # Receive the message in small chunks
        data = clientsocket.recv(1024)  # Gets the incoming message
        if data:
            # Decode the received data
            message = data.decode('utf-8')
            print(f'Received: {message}')
        else:
            print("No more data. Closing connection.")
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break
    time.sleep(5)  # Optional: sleep for 5 seconds

# Clean up the connection
clientsocket.close()