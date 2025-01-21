import socket as s

IP = s.gethostbyname(s.gethostname())
port = 4455
Addr = (IP, port)

def main():
    print("[Starting] Server is starting..")
    server = s.socket(s.AF_INET, s.SOCK_STREAM)
    server.bind(Addr)  
    server.listen()
    print(f"[Listening] Server is listening on {IP}:{port}")

    while True:
        conn, addr = server.accept()
        print(f"[New Connection] {addr} connected")

        filename = conn.recv(1024).decode('utf-8')
        print(filename)
        print("[RECV] Filename received")

        file = open(filename, "w")
        conn.send("Filename received.".encode('utf-8'))

        data = conn.recv(1024).decode('utf-8')
        print("[RECV] File data received")
        file.write(data)
        conn.send("File data received".encode('utf-8'))

        file.close()
        conn.close()

        print(f"[Disconnected] {addr} disconnected.")

if __name__ == "__main__":
    main()