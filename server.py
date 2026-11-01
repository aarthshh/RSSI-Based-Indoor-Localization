from flask import Flask, request, jsonify
from datetime import datetime, timezone
from threading import Lock

from config import ANCHORS

app = Flask(__name__)

# Shared state for the latest RSSI measurements
latest_measurements = {}
last_updated = None
data_lock = Lock()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "application": "RSSI Simulation",
        "status": "running",
        "message": "RSSI Simulation server is active"
    })


@app.route("/rssi", methods=["POST"])
def receive_rssi():
    global latest_measurements, last_updated

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Expected a JSON object"}), 400

    validated = {}

    # Validate measurements against known anchors
    for name in ANCHORS:
        if name not in data:
            return jsonify({"status": "error", "message": f"Missing RSSI value for {name}"}), 400

        val = data[name]
        
        # Ensure the value is a number (and strictly not a boolean) within a valid RSSI range
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not (-120 <= val <= 0):
            return jsonify({"status": "error", "message": f"Invalid RSSI value for {name}"}), 400

        validated[name] = float(val)

    # Safely update the global state
    with data_lock:
        latest_measurements = validated
        last_updated = datetime.now(timezone.utc).isoformat()

    return jsonify({
        "status": "success",
        "message": "RSSI measurements received",
        "measurements": validated,
        "timestamp": last_updated
    })


@app.route("/latest", methods=["GET"])
def get_latest_rssi():
    with data_lock:
        measurements = latest_measurements.copy()
        timestamp = last_updated

    if not measurements:
        return jsonify({
            "status": "waiting",
            "message": "No RSSI measurements received yet"
        })

    return jsonify({
        "status": "success",
        "measurements": measurements,
        "timestamp": timestamp
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
