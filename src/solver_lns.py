#!/usr/bin/env python3
import random
import time
import json
import pathlib
import sys
from copy import deepcopy
import streamlit as st

from utils.cvrp_parser import read_vrp
from utils.constructive_ls import savings_initial, route_length, two_opt

def remove_random(routes: list[list[int]], k: int, depot: int = 1) -> set[int]:
    """
    Видаляє k випадкових клієнтів із маршрутів.
    Повертає множину видалених клієнтів.
    """
    removed = set()
    while len(removed) < k:
        r = random.choice(routes)
        if len(r) <= 3:  # [depot, client, depot] — нічого не видаляємо
            continue
        idx = random.randint(1, len(r) - 2)
        removed.add(r.pop(idx))
        if len(r) == 2:  # маршрут спорожнів
            routes.remove(r)
    return removed

def cheapest_insert(
    routes: list[list[int]],
    customer: int,
    coords: dict[int, tuple[float, float]],
    demands: dict[int, int],
    cap: int,
    depot: int = 1
) -> None:
    """
    Вставляє customer у ту позицію серед усіх маршрутів, 
    яка дає мінімальне прирощення довжини.
    Якщо жодного підходящого маршруту немає, створює новий.
    """
    best_inc = float('inf')
    best_r = None
    best_pos = None

    for r in routes:
        load = sum(demands[i] for i in r)
        if load + demands[customer] > cap:
            continue
        for pos in range(1, len(r)):
            a, b = r[pos - 1], r[pos]
            inc = (dist(coords, a, customer)
                   + dist(coords, customer, b)
                   - dist(coords, a, b))
            if inc < best_inc:
                best_inc, best_r, best_pos = inc, r, pos

    if best_r is None:
        routes.append([depot, customer, depot])
    else:
        best_r.insert(best_pos, customer)

def dist(coords: dict[int, tuple[float, float]], i: int, j: int) -> float:
    """Евклідова відстань між двома вузлами."""
    xi, yi = coords[i]
    xj, yj = coords[j]
    return ((xi - xj)**2 + (yi - yj)**2)**0.5

def total_len(routes: list[list[int]], coords: dict[int, tuple[float, float]]) -> float:
    """Сума довжин усіх маршрутів."""
    return sum(route_length(r, coords) for r in routes)

def solve(
    file_path: str | pathlib.Path,
    sec_limit: int = 30,
    destroy_frac: float = 0.20
) -> dict:
    # 1) Прочитати інстанс
    data = read_vrp(file_path)
    cap     = data["capacity"]
    coords  = data["coords"]
    demands = data["demand"]
    depot   = data["depot_id"]
    depot = 1
    n_clients = len(coords) - 1

    # 2) Ініціалізувати рішення через Savings
    best_routes = savings_initial(cap, coords, demands, depot)
    best_len = total_len(best_routes, coords)

    cur_routes = deepcopy(best_routes)
    t0 = time.time()

    # 3) Основний цикл LNS
    while time.time() - t0 < sec_limit:
        if st.session_state.get("stop_flag"):  # ⛔ якщо користувач натиснув "Зупинити"
            break
        # RUIN
        k = max(1, int(destroy_frac * n_clients))
        new_routes = deepcopy(cur_routes)
        removed = remove_random(new_routes, k, depot)

        # RECREATE
        for cust in removed:
            cheapest_insert(new_routes, cust, coords, demands, cap, depot)

        # LOCAL IMPROVEMENT
        for r in new_routes:
            if len(r) > 4:
                two_opt(r, coords)

        new_len = total_len(new_routes, coords)

        # ACCEPTANCE: тільки покращення
        if new_len < best_len:
            best_routes, best_len = deepcopy(new_routes), new_len
            cur_routes = deepcopy(new_routes)

    # 4) Повернути результат
    return {
        "file": pathlib.Path(file_path).name,
        "algo": "LNS",
        "vehicles": len(best_routes),
        "distance": int(best_len),
        "time_sec": round(time.time() - t0, 2),
        "routes": best_routes
    }

if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/cvrplib/A-n32-k5.vrp"
    sec = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    result = solve(fp, sec)
    print(json.dumps(result, indent=2))
