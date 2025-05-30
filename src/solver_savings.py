# src/solver_savings.py
from __future__ import annotations
import time, pathlib, json
import numpy as np
from src.utils.cvrp_parser import read_vrp
import streamlit as st   


def _distance_matrix(coords: dict[int, tuple[float, float]]) -> np.ndarray:
    n = len(coords) + 1           # щоб індексувати напряму номером вузла (1-based)
    d = np.zeros((n, n), dtype=int)
    for i, (xi, yi) in coords.items():
        for j, (xj, yj) in coords.items():
            d[i, j] = int(round(((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5))
    return d


def solve(file_path: str, sec_limit: int | None = None) -> dict:
    """
    Clarke–Wright Savings для CVRP.
    sec_limit не потрібен — працює миттєво.
    """
    t0 = time.time()
    cap, coords, demands = read_vrp(str(file_path))
    depot = 1

    clients = [i for i in coords if i != depot]
    dist = _distance_matrix(coords)

    # ———————————————————————————————————
    # 1. Початкові маршрути (кожен клієнт окремо)
    routes: dict[int, list[int]] = {}
    load:   dict[int, int] = {}
    which:  dict[int, int] = {}         # клієнт → id маршруту

    for c in clients:
        routes[c] = [depot, c, depot]   # ключ == перший клієнт
        load[c]   = demands[c]
        which[c]  = c

    # ———————————————————————————————————
    # 2. Рахуємо savings
    savings: list[tuple[int, int, int]] = []
    for i in clients:
        for j in clients:
            if i < j:
                s = dist[depot, i] + dist[depot, j] - dist[i, j]
                savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda t: t[0])    # найбільші спочатку

    # ———————————————————————————————————
    # 3. Головне коло злиття
    for _, i, j in savings:
        ri, rj = which[i], which[j]
        if ri == rj:
            continue

        R_i = routes[ri]
        R_j = routes[rj]
        load_i, load_j = load[ri], load[rj]
        if load_i + load_j > cap:
            continue

        merged = None

        # i — хвіст R_i,  j — голова R_j
        if R_i[-2] == i and R_j[1] == j:
            merged = R_i[:-1] + R_j[1:]

        # i — голова R_i, j — хвіст R_j
        elif R_i[1] == i and R_j[-2] == j:
            merged = R_j[:-1] + R_i[1:]

        # обидва голови → треба перевернути R_i
        elif R_i[1] == i and R_j[1] == j:
            merged = R_i[::-1][:-1] + R_j[1:]

        # обидва хвости → перевертаємо R_j
        elif R_i[-2] == i and R_j[-2] == j:
            merged = R_i[:-1] + R_j[::-1][1:]

        if merged is None:
            continue

        # — злиття пройшло —
        new_key = min(ri, rj)
        # видаляємо старі
        for key in (ri, rj):
            routes.pop(key, None)
            load.pop(key, None)

        # додаємо новий
        routes[new_key] = merged
        load[new_key]   = load_i + load_j
        for node in merged:
            if node != depot:
                which[node] = new_key

    # ———————————————————————————————————
    # 4. Фінальний підрахунок
    final_routes = list(routes.values())
    total_dist = 0
    for r in final_routes:
        for a, b in zip(r[:-1], r[1:]):
            total_dist += dist[a, b]

    out = {
        "file": pathlib.Path(file_path).name,
        "algo": "Savings",
        "vehicles": len(final_routes),
        "distance": int(total_dist),
        "time_sec": round(time.time() - t0, 4),
        "routes": final_routes,
    }
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(solve(sys.argv[1]), indent=2))
