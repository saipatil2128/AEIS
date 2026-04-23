from scapy.all import sniff
import json
import time
import os

def isolate_device(ip):
    print("Isolating device:",ip)

    os.system(f'netsh advfirewall firewall add rule name="AEIS_BLOCK_IN_{ip}" dir=in action=block remoteip={ip}')
    os.system(f'netsh advfirewall firewall add rule name="AEIS_BLOCK_OUT_{ip}" dir=out action=block remoteip={ip}')

def restore_device(ip):
    print("Restoring network access for:", ip)

    os.system(f'netsh advfirewall firewall delete rule name="AEIS_BLOCK_IN_{ip}"')
    os.system(f'netsh advfirewall firewall delete rule name="AEIS_BLOCK_OUT_{ip}"')

packet_count = 0
THRESHOLD = 7000
sniff(timeout=2)

def packet_handler(packet):
    global packet_count
    packet_count += 1


print("AEIS monitoring started...")

while True:

    packet_count = 0   # reset counter

    sniff(
        timeout=5,
        prn=packet_handler,
        store=0,
        filter = "not port 4747"
    )

    packets = packet_count

    print("Packets in last interval:", packets)

    # Write traffic for dashboard
    traffic = {
        "packets": packets,
        "timestamp": time.time()
    }

    with open("dashboard/traffic.json", "w") as f:
        json.dump(traffic, f)

    status="NORMAL"
    threat = "LOW"

    device_ip = "192.168.29.83" 

    # Threat detection
    if packets >= THRESHOLD:

        status = "QUARANTINED"
        print("⚠ AEIS ALERT: Device quarantined")

        isolate_device(device_ip)

    else:

        status = "NORMAL"
        print("Status: NORMAL")

        restore_device(device_ip)

    alert = {
        "device": "Android Camera",
        "status": status,
        "threat": threat,
        "timestamp": time.time()
    }

    with open("dashboard/alert.json", "w") as f:
        json.dump(alert, f)

#cd C:\Users\naksh\OneDrive\Desktop\AIES_Project