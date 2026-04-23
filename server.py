from flask import Flask, request, jsonify
from flask_cors import CORS

import joblib
import numpy as np
import pandas as pd
import subprocess

from aeis_utils import BASE_FEATURES, engineer_features
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# 🔥 CHANGE THIS
PHONE_IP = "192.168.29.83"

# Load models
iso = joblib.load("outputs_if/model_isolation_forest.pkl")
threshold_if = float(np.load("outputs_if/iso_threshold.npy"))

rf = joblib.load("outputs_rf/model_random_forest.pkl")
threshold_rf = float(np.load("outputs_rf/rf_threshold.npy"))

traffic_data = {"packets": 0}
alert_data = {
    "device": "Android Camera",
    "status": "NORMAL",
    "threat": "LOW",
    "rf_prob": 0,
    "iso_score": 0
}

device_blocked = False


# 🔴 BLOCK DEVICE (used for demo + camera freeze)
def block_device():
    global device_blocked
    if device_blocked:
        return

    print("🚫 Blocking device:", PHONE_IP)

    try:
        subprocess.run(
            [
                "netsh",
                "advfirewall",
                "firewall",
                "add",
                "rule",
                'name=AEIS_Block',
                "dir=in",
                "action=block",
                f"remoteip={PHONE_IP}"
            ],
            check=True,
            capture_output=True
        )
        device_blocked = True
        print("✅ Firewall rule applied")

    except Exception as e:
        print("⚠ Firewall block failed:", e)


# 🔓 UNBLOCK DEVICE
def unblock_device():
    global device_blocked
    if not device_blocked:
        return

    print("🔓 Unblocking device:", PHONE_IP)

    try:
        subprocess.run(
            [
                "netsh",
                "advfirewall",
                "firewall",
                "delete",
                "rule",
                'name=AEIS_Block'
            ],
            check=True,
            capture_output=True
        )
        device_blocked = False
        print("✅ Firewall rule removed")

    except Exception as e:
        print("⚠ Firewall unblock failed:", e)


@app.route("/data", methods=["POST"])
def receive_data():
    global traffic_data, alert_data

    data = request.json

    if "packets_per_min" not in data:
        return jsonify({"ignored": True})

    packets = data["packets_per_min"]

    # 🔹 Prepare features
    raw = pd.DataFrame([[
        data["packets_per_min"],
        data["avg_packet_size"],
        data["activity_hour"],
        data["dest_count"]
    ]], columns=BASE_FEATURES)

    feat = engineer_features(raw)

    # 🔹 MODEL OUTPUTS
    iso_score = -iso.score_samples(feat.values)[0]
    rf_prob = rf.predict_proba(feat.values)[0, 1]

    iso_flag = iso_score >= threshold_if
    rf_flag = rf_prob >= threshold_rf

    # 🔥 MODEL-BASED BASE DECISION
    if iso_flag and rf_flag:
        status = "SUSPICIOUS"
        threat = "MEDIUM"
    elif iso_flag or rf_flag:
        status = "SUSPICIOUS"
        threat = "LOW"
    else:
        status = "NORMAL"
        threat = "LOW"

    # 🔥 ESCALATION USING TRAFFIC (NOT REPLACEMENT)
    if status != "NORMAL" and packets > 3000:
        status = "QUARANTINED"
        threat = "HIGH"
        block_device()

    elif status == "NORMAL" and packets < 1500:
        unblock_device()
    
    
    print("\n==============================", flush=True)
    print(f"Packets = {packets}", flush=True)
    print(f"RF Probability = {rf_prob:.4f}", flush=True)
    print(f"ISO Score = {iso_score:.4f}", flush=True)
    print(f"ISO Flag = {iso_flag}", flush=True)
    print(f"RF Flag = {rf_flag}", flush=True)
    print(f"FINAL STATUS = {status}", flush=True)
    print("==============================\n", flush=True)

    # 🔹 Update dashboard
    traffic_data = {"packets": packets}

    alert_data = {
        "device": "Android Camera",
        "status": status,
        "threat": threat,
        "rf_prob": round(float(rf_prob), 4),
        "iso_score": round(float(iso_score), 4)
    }

    # 🔥 DEBUG OUTPUT (SHOW THIS TO JUDGES)
    print(f"\nPackets={packets} | RF={rf_prob:.2f} | ISO={iso_score:.2f} | Status={status}\n")

    return jsonify({"ok": True})


@app.route("/traffic")
def traffic():
    return jsonify(traffic_data)


@app.route("/alert")
def alert():
    return jsonify(alert_data)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)