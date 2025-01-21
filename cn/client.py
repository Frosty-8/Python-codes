import socket as s

IP = s.gethostbyname(s.gethostname())
port = 4455
Addr = (IP, port)

def main():
    client = s.socket(s.AF_INET, s.SOCK_STREAM)
    client.connect(Addr)

    # Open the file and read its contents
    file = open("rt.txt", "r")
    data = file.read()

    # Send the filename to the server
    client.send("rt.txt".encode('utf-8'))
    msg = client.recv(1024).decode('utf-8')  # Receive confirmation from server
    print(f"[Server] : {msg}")

    # Send the file data to the server
    client.send(data.encode('utf-8'))  # Send the actual file data
    msg = client.recv(1024).decode('utf-8')  # Receive confirmation from server
    print(f"[Server] : {msg}")

    # Close the file and the client socket
    file.close()
    client.close()

if __name__ == "__main__":
    main()
