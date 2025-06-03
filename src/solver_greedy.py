# solver_greedy.py

import time
from utils.cvrp_parser import read_vrp
from math import sqrt


def euclidean(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def solve(filepath, time_limit=30):
    start = time.time()
    data = read_vrp(filepath)
    cap     = data["capacity"]
    coords  = data["coords"]
    demands = data["demand"]
    depot   = data["depot_id"]
    n = len(coords)
    customers = set(coords.keys()) - {1}  # 1 — depot
    unvisited = customers.copy()
    routes = []
    total_distance = 0.0

    while unvisited:
        route = [1]
        capacity_left = cap
        current = 1
        while True:
            candidates = [i for i in unvisited if demands[i] <= capacity_left]
            if not candidates:
                break
            # Вибрати найближчого клієнта
            next_customer = min(candidates, key=lambda i: euclidean(coords[current], coords[i]))
            route.append(next_customer)
            total_distance += euclidean(coords[current], coords[next_customer])
            capacity_left -= demands[next_customer]
            unvisited.remove(next_customer)
            current = next_customer
        route.append(1)
        total_distance += euclidean(coords[current], coords[1])
        routes.append(route)

    time_elapsed = time.time() - start
    return {
        "file": filepath.name,
        "vehicles": len(routes),
        "routes": routes,
        "distance": total_distance,
        "time_sec": round(time_elapsed, 2),
        "capacity": cap,
        "optimal": getattr(filepath, "optimal", None),  # опціонально
    }
