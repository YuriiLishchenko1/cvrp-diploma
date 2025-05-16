import streamlit as st
from pathlib import Path
import json, random, folium, numpy as np
from streamlit_folium import st_folium
from utils.cvrp_parser import read_vrp
from solver_baseline import solve

DATA_DIR = Path("data/cvrplib")

# ---------- Sidebar ----------
st.sidebar.title("CVRP-Diploma Demo")

bench_files = sorted(f.name for f in DATA_DIR.glob("*.vrp"))
file_choice = st.sidebar.selectbox("Benchmark", bench_files)
sec_limit   = st.sidebar.slider("Time limit OR-Tools, s", 5, 60, 30)

if st.sidebar.button("Solve"):
    result = solve(DATA_DIR / file_choice, sec_limit)
    st.session_state["res"] = result
else:
    result = st.session_state.get("res")

# ---------- Main ----------
st.title("CVRP baseline (OR-Tools + Folium)")

if result:
    st.subheader(f"📄 {result['file']}  |  🚚 {result['vehicles']} veh  |  🛣️ {result['distance']} dist")
    st.caption(f"⏱️ {result['time_sec']} s  |  FirstSolution=SAVINGS, GLS")

    # -------- Map --------
    cap, coords, _ = read_vrp(DATA_DIR / file_choice)
    latlng = {idx: (y, x) for idx, (x, y) in coords.items()}  # folium: (lat, lon)

    depot = 1
    m   = folium.Map(location=latlng[depot], zoom_start=12, tiles="CartoDB positron")
    folium.Marker(latlng[depot], icon=folium.Icon(color="red"), tooltip="Depot").add_to(m)

    colors = ["blue", "green", "purple", "orange", "cadetblue", "darkred",
              "darkblue", "darkgreen", "lightgray"]

    for ridx, route in enumerate(result["routes"]):
        col = colors[ridx % len(colors)]
        pts = [latlng[i] for i in route]
        folium.PolyLine(pts, color=col, weight=3, opacity=0.8).add_to(m)
        for i in route[1:-1]:  # клієнти
            folium.CircleMarker(latlng[i], radius=3, color=col, fill=True).add_to(m)

    st_folium(m, width=700, height=500)

    st.expander("JSON result").json(result)
else:
    st.info("Натисни **Solve** у сайдбарі, щоб згенерувати маршрути.")
