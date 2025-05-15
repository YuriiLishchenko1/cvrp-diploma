from utils.cvrp_parser import read_vrp
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np, time, sys, json, pathlib

def solve(file_path: str, sec_limit=30):
    cap, coords, demands = read_vrp(file_path)
    n, depot = len(coords), 1

    def dist(i, j):
        xi, yi = coords[i]; xj, yj = coords[j]
        return int(np.hypot(xi - xj, yi - yj))

    data = {
        "distance_matrix": [[dist(i, j) for j in coords] for i in coords],
        "demands": [demands[i] for i in coords],
        "vehicle_capacities": [cap] * n,
        "depot": depot - 1,
    }

    mgr = pywrapcp.RoutingIndexManager(n, n, data["depot"])
    routing = pywrapcp.RoutingModel(mgr)

    dist_cb = routing.RegisterTransitCallback(
        lambda i, j: data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb)

    demand_cb = routing.RegisterUnaryTransitCallback(
        lambda i: data["demands"][mgr.IndexToNode(i)]
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, data["vehicle_capacities"], True, "Capacity"
    )

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = sec_limit

    tic = time.time()
    sol = routing.SolveWithParameters(p)
    toc = time.time()
    if sol is None:
        raise RuntimeError("No solution")

    routes, total = [], 0
    for v in range(routing.vehicles()):
        idx = routing.Start(v)
        if routing.IsEnd(sol.Value(routing.NextVar(idx))):
            continue
        r = []
        while not routing.IsEnd(idx):
            r.append(mgr.IndexToNode(idx) + 1)
            idx = sol.Value(routing.NextVar(idx))
        r.append(depot)
        routes.append(r)
        total += sum(
            dist(a, b) for a, b in zip(r[:-1], r[1:])
        )

    return {"file": pathlib.Path(file_path).name,
            "vehicles": len(routes),
            "distance": total,
            "time_sec": round(toc - tic, 2),
            "routes": routes}

if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/cvrplib/A-n32-k5.vrp"
    res = solve(fp)
    print(json.dumps(res, indent=2))
