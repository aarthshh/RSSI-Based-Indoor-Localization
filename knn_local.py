import numpy as np
import json
from pathlib import Path

def predict_position_knn(measured_rssi, fingerprints, k=3):
    distances = []

    for coordinate, stored_rssi in fingerprints.items():

        # Compare measured RSSI with stored fingerprint
        difference = [
            measured_rssi[name] - stored_rssi[name]
            for name in measured_rssi
        ]

        rssi_distance = np.linalg.norm(difference)

        distances.append((rssi_distance, coordinate))

    # Find k most similar fingerprints
    distances.sort(key=lambda x: x[0])
    nearest = distances[:k]

    # Average their coordinates
    estimated_x = np.mean([point[0] for _, point in nearest])
    estimated_y = np.mean([point[1] for _, point in nearest])

    return np.array([estimated_x, estimated_y])

def load_real_fingerprints():
    file_path = Path(__file__).with_name(
        "real_fingerprints.json"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "real_fingerprints.json not found. "
            "Collect real fingerprints first."
        )

    with open(file_path, "r") as file:
        saved_data = json.load(file)

    fingerprints = {}

    for entry in saved_data.values():
        position = tuple(entry["position"])
        fingerprints[position] = entry["rssi"]

    if not fingerprints:
        raise ValueError(
            "No real fingerprints have been collected."
        )

    return fingerprints