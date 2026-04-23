from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route("/data", methods=["POST"])
def receive_data():
    data = request.json

    packets = data["packet_count"]
    avg_size = data["avg_packet_size"]

    # detection logic
    if packets > 15:
        status = "QUARANTINED"
        threat = "HIGH"
    else:
        status = "NORMAL"
        threat = "LOW"

    # save for dashboard
    with open("traffic.json", "w") as f:
        json.dump({"packets": packets}, f)

    with open("alert.json", "w") as f:
        json.dump({
            "status": status,
            "threat": threat
        }, f)

    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)