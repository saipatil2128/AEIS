from scapy.all import sniff, IP
import requests
import time

LAPTOP1_URL = "http://192.168.29.151:5000/data"  # CHANGE if needed

packet_data = []

def packet_handler(packet):
    if IP in packet:
        packet_data.append({
            "src": packet[IP].src,
            "dst": packet[IP].dst,
            "len": len(packet)
        })

def extract_features(data):
    if len(data) == 0:
        return None

    packet_count = len(data)
    avg_size = sum(p["len"] for p in data) / packet_count
    unique_dest = len(set(p["dst"] for p in data))

    return {
        "packets_per_min": packet_count,
        "avg_packet_size": avg_size,
        "activity_hour": time.localtime().tm_hour,
        "dest_count": unique_dest
    }

print("🚀 AEIS Live Pipeline Started...")

while True:
    packet_data.clear()

    sniff(timeout=5, prn=packet_handler)

    features = extract_features(packet_data)

    if features:
        print("📡 Sending:", features)

        try:
            requests.post(LAPTOP1_URL, json=features)
        except:
            print("❌ Connection error...")