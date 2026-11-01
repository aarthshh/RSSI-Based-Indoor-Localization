import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import requests
from datetime import datetime, timezone

from config import ANCHORS, AREA_WIDTH, AREA_HEIGHT
from localization import calibrated_rssi_to_distance
from multilateration import estimate_position
from knn_local import load_real_fingerprints, predict_position_knn


# Page configuration & defaults
st.set_page_config(page_title="RSSI Simulation", page_icon="📡", layout="wide")

DEFAULT_SERVER_URL = "http://127.0.0.1:5000"

st.title("RSSI Simulation")
st.caption("Real-Time Wi-Fi Based Localization System")


def fetch_real_rssi(server_url):
    """Fetch latest anchor RSSI readings from the Flask collector."""
    try:
        # Avoid routing local addresses through system HTTP proxies
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(f"{server_url.rstrip('/')}/latest", timeout=(3, 5))
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "success":
            return None, payload.get("message", "No RSSI data available yet")

        readings = payload.get("measurements")
        if not isinstance(readings, dict):
            return None, "Server returned no valid measurements dictionary"

        validated = {}
        for name in ANCHORS:
            if name not in readings:
                return None, f"Missing RSSI value for {name}"
            value = float(readings[name])
            if not np.isfinite(value) or not (-120 <= value <= 0):
                return None, f"Invalid RSSI value for {name}: {value}"
            validated[name] = value

        return {
            "measurements": validated,
            "timestamp": payload.get("timestamp"),
        }, None

    except (requests.RequestException, ValueError, TypeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


# Sidebar Controls 
st.sidebar.header("RSSI Simulation Controls")

server_url = st.sidebar.text_input(
    "Flask RSSI Server URL",
    value=DEFAULT_SERVER_URL,
    help="Use an address reachable from this environment."
).strip().rstrip("/")
st.sidebar.caption("Flask uses port 5000; Streamlit uses port 8501.")

algorithm = st.sidebar.selectbox(
    "Localization Algorithm",
    ["Calibrated Multilateration", "KNN Fingerprinting"]
)

k = 3
if algorithm == "KNN Fingerprinting":
    k = st.sidebar.slider("KNN Neighbors", min_value=1, max_value=10, value=3)

auto_refresh = st.sidebar.toggle("Automatic Location Updates", value=True)
if not auto_refresh and st.sidebar.button("Refresh Location"):
    st.rerun()

st.sidebar.caption("Automatic mode checks for new RSSI data every 5 seconds.")


@st.cache_data
def get_real_fingerprints():
    return load_real_fingerprints()


def calculate_distances(measured_rssi):
    """Convert RSSI readings to distance estimates using log-distance path loss."""
    return {
        name: calibrated_rssi_to_distance(measured_rssi[name], name)
        for name in ANCHORS
    }


def localize(measured_rssi, algorithm, k):
    """Estimate 2D coordinates using multilateration or KNN fingerprinting."""
    if algorithm == "Calibrated Multilateration":
        distances = calculate_distances(measured_rssi)
        estimated_position = estimate_position(ANCHORS, distances)
        return estimated_position, distances, None

    fingerprints = get_real_fingerprints()
    if not fingerprints:
        raise ValueError("Real fingerprint database is empty")

    estimated_position = predict_position_knn(
        measured_rssi,
        fingerprints,
        k=min(k, len(fingerprints))
    )
    return estimated_position, None, fingerprints


# Live Dashboard View 
@st.fragment(run_every="5s" if auto_refresh else None)
def live_dashboard():
    real_data, fetch_error = fetch_real_rssi(server_url)

    if real_data is None:
        st.error("Cannot read RSSI from the Flask server")
        st.code(f"{server_url}/latest", language="text")
        st.warning(fetch_error or "No measurements available")
        st.info(
            f"Test connection via: curl --noproxy '*' --max-time 10 '{server_url}/latest'. "
            "If localhost fails, update the sidebar URL to the host IP (e.g. http://10.x.x.x:5000)."
        )
        return

    measured_rssi = real_data["measurements"]
    timestamp = real_data["timestamp"]

    # Check measurement freshness
    measurement_age = None
    if timestamp:
        try:
            received_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if received_time.tzinfo is None:
                raise ValueError("Missing timezone in timestamp")
            measurement_age = (datetime.now(timezone.utc) - received_time).total_seconds()
        except ValueError:
            measurement_age = None

    if measurement_age is None:
        st.warning("Measurement timestamp unavailable or invalid.")
    elif measurement_age < 0:
        st.warning("Measurement timestamp is in the future.")
    elif measurement_age > 30:
        st.warning(f"Measurements are {measurement_age:.0f} seconds old. Displaying the last estimate.")
    else:
        st.success("Real RSSI measurements received.")

    # Calculate position
    try:
        estimated_position, distances, fingerprints = localize(measured_rssi, algorithm, k)
    except (FileNotFoundError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        st.error(f"Localization error: {error}")
        return

    estimated_x = float(estimated_position[0])
    estimated_y = float(estimated_position[1])

    if not (np.isfinite(estimated_x) and np.isfinite(estimated_y)):
        st.error("Localization produced invalid coordinates.")
        return

    # Visual display & summary metrics
    map_column, stats_column = st.columns([3, 1], gap="large")

    with stats_column:
        st.subheader("Tracking Status")
        if measurement_age is None or measurement_age < 0:
            st.warning("Timestamp Invalid")
        elif measurement_age > 30:
            st.error("Stale Measurements")
        else:
            st.success("Measurements Available")

        st.metric("Estimated X", f"{estimated_x:.2f} m")
        st.metric("Estimated Y", f"{estimated_y:.2f} m")
        st.metric("Active Anchors", len(ANCHORS))
        if measurement_age is not None:
            st.metric("Data Age", f"{max(0, measurement_age):.1f} s")

    with map_column:
        st.subheader("Localization Map")
        fig, ax = plt.subplots(figsize=(10, 7))

        # Anchor markers
        for name, (anchor_x, anchor_y) in ANCHORS.items():
            ax.scatter(anchor_x, anchor_y, color="#2563EB", marker="^", s=160, zorder=5)
            ax.annotate(
                name,
                (anchor_x, anchor_y),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold"
            )

        # Estimated target position
        ax.scatter(
            estimated_x,
            estimated_y,
            color="#EF4444",
            marker="o",
            s=200,
            edgecolors="white",
            linewidths=2,
            label="Estimated Android Position",
            zorder=6
        )
        ax.annotate(
            f"({estimated_x:.1f}, {estimated_y:.1f})",
            (estimated_x, estimated_y),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=10
        )

        ax.set_xlim(-2, AREA_WIDTH + 2)
        ax.set_ylim(-2, AREA_HEIGHT + 2)
        ax.set_aspect("equal")
        ax.set_xlabel("X Coordinate (meters)")
        ax.set_ylabel("Y Coordinate (meters)")
        ax.set_title(f"RSSI Simulation — {algorithm}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

        st.pyplot(fig)
        plt.close(fig)

    # RSSI measurements table
    st.subheader("Real RSSI Measurements")
    rssi_table = []
    for name in ANCHORS:
        row = {
            "Anchor": name,
            "RSSI (dBm)": round(measured_rssi[name], 2)
        }
        if distances is not None:
            row["Estimated Distance (m)"] = round(float(distances[name]), 2)
        rssi_table.append(row)

    st.dataframe(rssi_table, use_container_width=True, hide_index=True)

    # System & algorithm details
    st.subheader("System Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Localization Method", algorithm)
    col2.metric("Reference Devices", len(ANCHORS))
    col3.metric("Data Source", "Real Android RSSI")

    if fingerprints is not None:
        st.subheader("Fingerprinting Information")
        fcol1, fcol2 = st.columns(2)
        fcol1.metric("Stored Fingerprints", len(fingerprints))
        fcol2.metric("KNN Neighbors", min(k, len(fingerprints)))

    st.caption(
        "RSSI Simulation uses real Wi-Fi RSSI measurements. "
        "Position estimates depend on calibration, anchor placement, and fingerprint quality."
    )


# Run dashboard fragment
live_dashboard()
