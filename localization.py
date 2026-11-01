# Estimates distance from RSSI, accounting for wall as well.
def rssi_to_distance_wall_aware(rssi, wall_count, rssi_at_1=-40, path_loss=2.2, wall_loss=5):
    total_wall_loss = wall_count*wall_loss
    d = 10**((rssi_at_1 - rssi - total_wall_loss)/(10*path_loss))

    return d

# Estimates distance from RSSI using basic path loss model.
def rssi_to_distance(rssi, rssi_at_1 = -40, path_loss = 2.2):
    d = 10**((rssi_at_1 - rssi) / (10*path_loss))

    return d

import json
from pathlib import Path

import json
import math
from pathlib import Path

CALIBRATION_FILE = Path(__file__).with_name("calibration.json")


def calibrated_rssi_to_distance(rssi, anchor_name):

    with open(CALIBRATION_FILE, "r") as file:
        calibration = json.load(file)

    params = calibration[anchor_name]

    rssi_at_1m = float(params["rssi_at_1m"])
    path_loss = float(params["path_loss"])
    rssi = float(rssi)

    if not all(map(math.isfinite, [
        rssi_at_1m, path_loss, rssi
    ])):
        raise ValueError(
            f"Invalid calibration values for {anchor_name}"
        )

    if path_loss <= 0.1:
        raise ValueError(
            f"{anchor_name}: suspicious path_loss={path_loss:.6f}. "
            "Check your calibration measurements."
        )

    exponent = (
        (rssi_at_1m - rssi) / (10 * path_loss)
    )

    if exponent > 6 or exponent < -6:
        raise ValueError(
            f"{anchor_name}: unrealistic RSSI-derived distance. "
            f"Exponent={exponent:.2f}, "
            f"path_loss={path_loss:.4f}. "
            "Check calibration data."
        )

    distance = 10 ** exponent

    return distance