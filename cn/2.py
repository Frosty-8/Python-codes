import socket as s

ob = s.socket()
ob.bind(('localhost',2301))

ob.listen(1)
print("Server is ready to accept connection")
clientobj,add=ob.accept()

print("Connected with this address",add)

conn=True
while conn:
    gotmsg = clientobj.recv(1024)
    gotmsg.decode('utf-8')
    print(gotmsg)
    if(len(gotmsg)==0):
        conn = False
        ob.close()  