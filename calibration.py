# calibration.py - RSSI Simulation Calibration

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from config import ANCHORS

CALIBRATION_FILE = Path(__file__).with_name("calibration.json")
WAP_NAMES = {"A1": "WAP_1", "A2": "WAP_2", "A3": "WAP_3"}


def load_calibration():
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return {}


def save_calibration(anchor_name, results):
    calibration = load_calibration()
    calibration[anchor_name] = results
    
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calibration, f, indent=4)
        
    print(f"\nSaved calibration results to {CALIBRATION_FILE}")


def collect_measurements():
    distances = []
    rssi_means = []

    print("\nProvide measurements at known distances (e.g., -45,-47,-46). Type 'done' to finish.\n")

    while True:
        user_input = input("Distance in meters (or 'done'): ").strip()
        if user_input.lower() == "done":
            break

        try:
            distance = float(user_input)
            if not np.isfinite(distance) or distance <= 0:
                print("-> Please enter a positive number.")
                continue

            if distance in distances:
                print("-> Already recorded this distance.")
                continue

            readings_input = input(f"RSSI readings at {distance:g}m: ")
            readings = [float(val.strip()) for val in readings_input.split(",")]

            if not readings or not all(np.isfinite(v) and -120 <= v <= 0 for v in readings):
                print("-> Invalid RSSI values. Must be between -120 and 0.")
                continue

            mean_rssi = float(np.mean(readings))
            distances.append(distance)
            rssi_means.append(mean_rssi)

            print(f"-> Logged: {distance:g}m -> {mean_rssi:.2f} dBm ({len(readings)} readings)\n")

        except ValueError:
            print("-> Invalid input format. Try again.\n")

    return distances, rssi_means


def fit_path_loss_model(distances, rssi_values):
    if len(distances) < 3:
        raise ValueError("Need at least 3 points for curve fitting.")

    distances = np.array(distances, dtype=float)
    rssi_values = np.array(rssi_values, dtype=float)
    log_distances = np.log10(distances)

    # Log-distance path loss model: RSSI = A - 10*n*log10(d)
    slope, intercept = np.polyfit(log_distances, rssi_values, 1)

    rssi_at_1m = float(intercept)
    path_loss = float(-slope / 10)

    predicted_rssi = rssi_at_1m - (10 * path_loss * log_distances)
    rmse = float(np.sqrt(np.mean((rssi_values - predicted_rssi) ** 2)))

    return {
        "rssi_at_1m": rssi_at_1m,
        "path_loss": path_loss,
        "rmse_db": rmse
    }


def calibrate_anchor(anchor_name):
    hotspot = WAP_NAMES.get(anchor_name, anchor_name)
    print(f"\n--- Calibrating {anchor_name} ({hotspot}) ---")

    distances, rssi_values = collect_measurements()
    if len(distances) < 3:
        print("\nCalibration aborted: Not enough data points.")
        return

    try:
        results = fit_path_loss_model(distances, rssi_values)
    except Exception as e:
        print(f"\nError fitting model: {e}")
        return

    if results["path_loss"] <= 0:
        print("\nWarning: Path loss exponent is <= 0 (RSSI didn't drop with distance). Discarding.")
        return

    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    results["distances_m"] = distances
    results["mean_rssi_dbm"] = rssi_values

    print("\n--- Results ---")
    print(f"RSSI @ 1m:  {results['rssi_at_1m']:.2f} dBm")
    print(f"Path loss (n): {results['path_loss']:.3f}")
    print(f"RMSE:          {results['rmse_db']:.2f} dB")

    if input("\nSave calibration? (y/n): ").strip().lower() == "y":
        save_calibration(anchor_name, results)
    else:
        print("Discarded.")


def main():
    anchor_names = list(ANCHORS)

    while True:
        print("\n=== RSSI Simulation Calibration ===")
        for i, name in enumerate(anchor_names, 1):
            print(f"{i}. Calibrate {name} ({WAP_NAMES.get(name, name)})")
        print("4. View saved calibrations")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice in ("1", "2", "3"):
            idx = int(choice) - 1
            if idx < len(anchor_names):
                calibrate_anchor(anchor_names[idx])
            else:
                print("Invalid anchor selection.")
        elif choice == "4":
            calibrations = load_calibration()
            print(json.dumps(calibrations, indent=4) if calibrations else "\nNo calibrations found.")
        elif choice == "0":
            print("\nExiting.")
            break
        else:
            print("Invalid option, try again.")


if __name__ == "__main__":
    main()
