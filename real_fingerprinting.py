import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from config import ANCHORS, AREA_WIDTH, AREA_HEIGHT

SERVER_URL = "http://10.50.22.200:5000"
FINGERPRINT_FILE = Path(__file__).with_name("real_fingerprints.json")

N_SAMPLES = 5
SAMPLE_INTERVAL = 5


def load_fingerprints():
    if not FINGERPRINT_FILE.exists():
        return {}
    with open(FINGERPRINT_FILE, "r") as file:
        return json.load(file)


def save_fingerprints(fingerprints):
    with open(FINGERPRINT_FILE, "w") as file:
        json.dump(fingerprints, file, indent=4)
    print("Fingerprint database saved successfully.")


def fetch_rssi():
    """Fetch and validate latest anchor RSSI readings from the collector."""
    try:
        response = requests.get(f"{SERVER_URL}/latest", timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            print("No RSSI measurements available.")
            return None

        readings = data.get("measurements", {})
        if not all(name in readings for name in ANCHORS):
            print("Missing one or more anchor readings.")
            return None

        timestamp_text = data.get("timestamp")
        if not timestamp_text:
            print("Measurement timestamp missing.")
            return None

        # Check timestamp timezone and freshness (<30s)
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            print("Measurement timestamp has no timezone.")
            return None

        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age < 0 or age > 30:
            print("RSSI data is stale or has an invalid timestamp.")
            return None

        rssi = {}
        for name in ANCHORS:
            value = float(readings[name])
            if not math.isfinite(value) or not (-120 <= value <= 0):
                print(f"Invalid RSSI for {name}: {value}")
                return None
            rssi[name] = value

        return {"rssi": rssi, "timestamp": timestamp_text}

    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        print(f"Error fetching RSSI: {error}")
        return None


def collect_samples(n_samples=N_SAMPLES):
    """Collect n fresh RSSI samples at the current location and return their mean."""
    samples = {name: [] for name in ANCHORS}
    last_timestamp = None

    print("\nCollecting RSSI measurements...")
    print("Keep your Android phone at the known position.")
    print("Run send_rssi.sh on the phone to submit fresh scans.")

    attempts = 0
    max_attempts = n_samples * 12
    first_anchor = next(iter(ANCHORS))

    while len(samples[first_anchor]) < n_samples:
        if attempts >= max_attempts:
            print("Collection timed out.")
            break
        attempts += 1

        measurement = fetch_rssi()
        if measurement is not None:
            timestamp = measurement["timestamp"]
            # record new scans based on timestamp change
            if timestamp != last_timestamp:
                last_timestamp = timestamp
                for name in ANCHORS:
                    samples[name].append(measurement["rssi"][name])

                count = len(samples[first_anchor])
                print(f"Sample {count}/{n_samples}: {measurement['rssi']}")

        if len(samples[first_anchor]) < n_samples:
            time.sleep(SAMPLE_INTERVAL)

    count = len(samples[first_anchor])
    if count < n_samples:
        print(f"Only {count}/{n_samples} fresh submissions collected.")
        return None

    return {name: float(np.mean(values)) for name, values in samples.items()}


def record_fingerprint():
    """Prompt for known coordinates, collect RSSI samples, and save fingerprint."""
    try:
        x = float(input("\nEnter X coordinate (m): "))
        y = float(input("Enter Y coordinate (m): "))
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= AREA_WIDTH and 0 <= y <= AREA_HEIGHT):
            print(f"Coordinates must be within area bounds (0-{AREA_WIDTH}m, 0-{AREA_HEIGHT}m).")
            return
    except ValueError:
        print("Invalid coordinate input.")
        return

    print(f"\nRecording fingerprint at ({x}, {y})")
    mean_rssi = collect_samples()
    if mean_rssi is None:
        print("Fingerprint not saved.")
        return

    fingerprints = load_fingerprints()
    position_key = f"{x:g},{y:g}"
    fingerprints[position_key] = {
        "position": [x, y],
        "rssi": mean_rssi,
        "samples": N_SAMPLES,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    save_fingerprints(fingerprints)
    print("\nFingerprint recorded:")
    print(json.dumps(fingerprints[position_key], indent=4))


def view_fingerprints():
    """Display all stored location fingerprints."""
    fingerprints = load_fingerprints()
    if not fingerprints:
        print("No fingerprints saved yet.")
        return

    print(f"\nTotal fingerprints: {len(fingerprints)}")
    for key, value in fingerprints.items():
        print(f"Position {key}: {value['rssi']}")


def main():
    while True:
        print("\n========== RSSI SIMULATION FINGERPRINTING ==========")
        print("1. Record fingerprint at known position")
        print("2. View saved fingerprints")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()
        if choice == "1":
            record_fingerprint()
        elif choice == "2":
            view_fingerprints()
        elif choice == "0":
            print("Exiting fingerprint collection.")
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
