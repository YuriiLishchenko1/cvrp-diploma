from __future__ import annotations

"""
Large-Neighbourhood Search + Simulated Annealing для CVRP.
Запуск окремо:
    python -m src.solver_lns data/cvrplib/A-n32-k5.vrp 30
Алгоритм:
    1. стартова побудова    →   Clarke-Wright Savings
    2. цикл до time-limit:
           • destroy: 20 % випадкових клієнтів
           • repair : cheapest-insertion
           • accept : SA-критерій
    3. локальна 2-opt на кращому рішенні
    4. повертає json, сумісний з іншими solver-ами
"""

import random, math, time, json, pathlib, sys, re
from copy import deepcopy
from collections import defaultdict
import streamlit as st
from utils.cvrp_parser   import read_vrp
from utils.constructive_ls import savings_initial, route_length, two_opt, inter_route_swap, inter_route_relocate



# ────────────────────────────────────────────────────────────
def remove_random(routes, k:int, depot:int=1):
    """видаляє k випадкових клієнтів; повертає set вилучених"""
    removed = set()
    while len(removed) < k:
        r = random.choice(routes)
        if len(r) <= 3:     # depot-A-depot  – нічого прибирати
            continue
        idx = random.randint(1, len(r)-2)
        removed.add(r.pop(idx))
        if len(r) == 2:     # маршрут спорожнів
            routes.remove(r)
    return removed


def cheapest_insert(routes, customer, coords, demands,
                    cap, depot=1):
    """вставляє client у позицію з мінімальним збільшенням довжини"""
    best_inc, best_r, best_pos = math.inf, None, None
    for r in routes:
        load = sum(demands[i] for i in r if i != depot)  # ∑ demand у маршруті
        if load + demands[customer] > cap:
            continue
        for pos in range(1, len(r)):
            a, b = r[pos-1], r[pos]
            inc = (dist(coords,a,customer)+dist(coords,customer,b)
                   - dist(coords,a,b))
            if inc < best_inc:
                best_inc, best_r, best_pos = inc, r, pos
    # якщо не знайшли – створюємо новий маршрут
    if best_r is None:
        best_r = [depot, customer, depot]
        routes.append(best_r)
    else:
        best_r.insert(best_pos, customer)


def dist(c,x,y):
    c1, c2 = c[x], c[y]
    return math.hypot(c1[0]-c2[0], c1[1]-c2[1])


def total_len(routes, coords):
    return sum(route_length(r, coords) for r in routes)


def extract_max_vehicles(file_path: str) -> int | None:
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("COMMENT") and "No of trucks:" in line:
                    match = re.search(r"No of trucks:\s*(\d+)", line)
                    if match:
                        return int(match.group(1))
    except Exception:
        pass
    return None


# ────────────────────────────────────────────────────────────
def solve(file_path: str | pathlib.Path,
          sec_limit: int = 30,
          destroy_frac: float = .20,
          T0: float = 1000,
          cooling: float = .995,
          max_vehicles: int | None = None):

    data = read_vrp(file_path)
    cap     = data["capacity"]
    coords  = data["coords"]
    demands = data["demand"]
    depot   = data["depot_id"]
    depot = 1

    if max_vehicles is None:
        max_vehicles = data.get("num_vehicles")
    print("🛻 Num vehicles from parser:", data.get("num_vehicles"))

    
    best_routes = savings_initial(cap, coords, demands, depot, max_vehicles=max_vehicles)

    best_len    = total_len(best_routes, coords)

    cur_routes  = deepcopy(best_routes)
    cur_len     = best_len
    T           = T0

    start = time.time()
    while time.time() - start < sec_limit:
        if st.session_state.get("stop_flag"):
            break

        k = max(1, int(destroy_frac * (len(coords) - 1)))
        new_routes = deepcopy(cur_routes)
        removed = remove_random(new_routes, k, depot)

        for cust in removed:
            cheapest_insert(new_routes, cust, coords, demands, cap, depot)

        new_len = total_len(new_routes, coords)
        
        # ❗ Strict check after insertion
        if max_vehicles is not None and len(new_routes) > max_vehicles:
            penalty = (len(new_routes) - max_vehicles) * 10000
            new_len += penalty
        
        if new_len < cur_len or random.random() < math.exp(-(new_len - cur_len) / T):
            cur_routes, cur_len = new_routes, new_len
            if new_len < best_len:
                best_routes, best_len = deepcopy(new_routes), new_len

            cur_len = total_len(cur_routes, coords)
            if cur_len < best_len:
                best_routes, best_len = deepcopy(cur_routes), cur_len

        T *= cooling

    for r in best_routes:
        if len(r) > 4:
            two_opt(r, coords)
    best_len = total_len(best_routes, coords)


    return {
        "file": pathlib.Path(file_path).name,
        "algo": "LNS-SA",
        "vehicles": len(best_routes),
        "distance": int(best_len),
        "time_sec": round(time.time() - start, 2),
        "routes": best_routes
    }


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fp  = sys.argv[1] if len(sys.argv) > 1 else "data/cvrplib/A-n32-k5.vrp"
    sec = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(json.dumps(solve(fp, sec), indent=2))
