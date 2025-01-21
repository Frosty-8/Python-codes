import socket as s

ob=s.socket()

ob.connect(('localhost',2301))

conn=True
while conn:
    msg = input("Enter your message : ")
    ob.send(msg.encode('utf-8'))
    if msg=='no':
        conn=False
        ob.close()