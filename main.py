import numpy as np

from config import ANCHORS, target, walls
from simulator import simulate_rssi
from geometry import count_walls
from localization import rssi_to_distance_wall_aware
from multilateration import (estimate_position,estimate_position_weighted,position_error)
from fingerprinting import generate_grid, generate_fingerprints
from knn_local import predict_position_knn

n_runs = 30
n_samples = 15

errors = {
    "single": [],
    "mean": [],
    "weighted": [],
    "knn": []
}

# Generate fingerprint database once
grid = generate_grid(30, 20, 2)
fingerprints = generate_fingerprints(grid, ANCHORS, walls, n_samples=15)

for run in range(n_runs):
    samples_by_anchor = {}
    mean_rssi = {}
    weights = {}

    for name, anchor in ANCHORS.items():
        samples = [
            simulate_rssi(anchor, target, walls)
            for _ in range(n_samples)
        ]

        samples_by_anchor[name] = samples
        mean_rssi[name] = np.mean(samples)
        weights[name] = 1 / (np.var(samples) + 1e-6)

    single_distances = {}
    mean_distances = {}

    for name, anchor in ANCHORS.items():
        wall_count = count_walls(anchor, target, walls)

        single_distances[name] = rssi_to_distance_wall_aware(samples_by_anchor[name][0], wall_count)

        mean_distances[name] = rssi_to_distance_wall_aware(mean_rssi[name], wall_count)

    single_position = estimate_position(ANCHORS, single_distances)
    mean_position = estimate_position(ANCHORS, mean_distances)

    weighted_position = estimate_position_weighted(ANCHORS, mean_distances, weights)

    knn_position = predict_position_knn(mean_rssi, fingerprints, k=3)

    errors["single"].append(position_error(target, single_position))
    errors["mean"].append(position_error(target, mean_position))
    errors["weighted"].append(position_error(target, weighted_position))
    errors["knn"].append(position_error(target, knn_position))


for name, values in errors.items():
    print(f"\n{name.upper()}")
    print(f"Average error: {np.mean(values):.3f} m")
    print(f"Standard deviation: {np.std(values):.3f} m")