import numpy as np
from simulator import simulate_rssi
from config import ANCHORS, walls


def generate_grid(width, height, spacing):
    grid = []

    for x in range(0, width + 1, spacing):
        for y in range(0, height + 1, spacing):
            grid.append((x, y))

    return grid


def generate_fingerprints(grid, anchors, walls, n_samples=15):
    fingerprints = {}

    for coordinate in grid:

        if coordinate in anchors.values():
            continue

        rssi_values = {}

        for name, anchor in anchors.items():
            samples = []

            for i in range(n_samples):
                rssi = simulate_rssi(anchor, coordinate, walls)
                samples.append(rssi)

            mean_rssi = np.mean(samples)
            rssi_values[name] = mean_rssi

        fingerprints[coordinate] = rssi_values

    return fingerprints


grid = generate_grid(30, 20, 2)

fingerprints = generate_fingerprints(grid, ANCHORS, walls, n_samples=15)

print("Total fingerprint points:", len(grid))
print("Total fingerprints:", len(fingerprints))
print("Fingerprint at (2, 2):", fingerprints[(2, 2)])

print("\nFirst 5 fingerprints:")
for coordinate, readings in list(fingerprints.items())[:5]:
    print(coordinate, ":", readings)