# src/solver_baseline.py

import pathlib, sys, time, json
import numpy as np
from utils.cvrp_parser import read_vrp
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import streamlit as st

# Має приймати file_path, sec_limit та fs_strategy
def solve(file_path: str, sec_limit: int = 30, fs_strategy: str = "SAVINGS"):
    # Парсимо файл
    data = read_vrp(file_path)
    cap     = data["capacity"]
    coords  = data["coords"]
    demands = data["demand"]
    depot   = data["depot_id"]

    n, depot = len(coords), 1

    # Попередньо обчислюємо відстані
    def dist(i, j):
        xi, yi = coords[i]
        xj, yj = coords[j]
        return int(np.hypot(xi - xj, yi - yj))

    data = {
        "distance_matrix": [[dist(i, j) for j in coords] for i in coords],
        "demands": [demands[i] for i in coords],
        "vehicle_capacities": [cap] * n,
        "depot": depot - 1
    }

    mgr = pywrapcp.RoutingIndexManager(n, n, data["depot"])
    routing = pywrapcp.RoutingModel(mgr)

    # Встановлюємо функцію вартості дуги
    transit_cb = routing.RegisterTransitCallback(
        lambda i, j: data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Вимоги за вантажем
    demand_cb = routing.RegisterUnaryTransitCallback(
        lambda i: data["demands"][mgr.IndexToNode(i)]
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, data["vehicle_capacities"], True, "Capacity"
    )

    # Налаштовуємо стратегію початкового рішення
    strat_map = {
        "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "AUTOMATIC": routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        # за потреби додайте ще...
    }
    # Якщо користувач передав невідому стратегію — беремо AUTOMATIC
    fs_enum = strat_map.get(fs_strategy.upper(), routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = fs_enum
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = sec_limit

    t0 = time.time()
    sol = routing.SolveWithParameters(search_params)
    t1 = time.time()

    if sol is None:
        raise RuntimeError("No solution found within the time limit")

    routes, total = [], 0
    for v in range(n):
        if st.session_state.get("stop_flag"):  # ⛔ якщо користувач натиснув "Зупинити"
            break
        idx = routing.Start(v)
        # якщо цей транспортний засіб нікого не обслуговує — пропускаємо
        if routing.IsEnd(sol.Value(routing.NextVar(idx))):
            continue
        r = []
        while not routing.IsEnd(idx):
            r.append(mgr.IndexToNode(idx) + 1)  # +1 — щоб узгодити з вашим парсером
            idx = sol.Value(routing.NextVar(idx))
        r.append(depot)
        # рахуємо довжину в маршруті
        total += sum(dist(a, b) for a, b in zip(r[:-1], r[1:]))
        routes.append(r)

    return {
        "file": pathlib.Path(file_path).name,
        "vehicles": len(routes),
        "distance": total,
        "time_sec": round(t1 - t0, 2),
        "routes": routes
    }


if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/cvrplib/A-n32-k5.vrp"
    sec = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    strat = sys.argv[3] if len(sys.argv) > 3 else "SAVINGS"
    print(json.dumps(solve(fp, sec, strat), indent=2))
