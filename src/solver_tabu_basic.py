# solver_tabu_basic.py
import random, time, math, pathlib
from utils.cvrp_parser import read_vrp

def decode(chrom, demands, cap, depot=1):
    routes, load, r = [], 0, [depot]
    for n in chrom:
        if load + demands[n] <= cap:
            r.append(n)
            load += demands[n]
        else:
            r.append(depot)
            routes.append(r)
            r, load = [depot, n], demands[n]
    r.append(depot)
    routes.append(r)
    return routes

def route_length(route, coords):
    return sum(math.hypot(coords[route[i+1]][0] - coords[route[i]][0],
                          coords[route[i+1]][1] - coords[route[i]][1])
               for i in range(len(route)-1))

def total_cost(chrom, coords, demands, cap, depot=1):
    routes = decode(chrom, demands, cap, depot)
    return sum(route_length(r, coords) for r in routes), routes

def solve(file_path, sec_limit=30, tabu_tenure=10):
    data = read_vrp(file_path)
    cap     = data["capacity"]
    coords  = data["coords"]
    demands = data["demand"]
    depot   = data["depot_id"]
    depot = 1
    clients = [i for i in coords if i != depot]
    current = clients[:]
    random.shuffle(current)

    best = current[:]
    best_cost, best_routes = total_cost(best, coords, demands, cap, depot)
    tabu_list = []
    start = time.time()

    while time.time() - start < sec_limit:
        neighbors = []
        for i in range(len(current) - 1):
            for j in range(i + 1, len(current)):
                if (i, j) in tabu_list or (j, i) in tabu_list:
                    continue
                neighbor = current[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                cost, _ = total_cost(neighbor, coords, demands, cap, depot)
                neighbors.append((cost, neighbor, (i, j)))

        if not neighbors:
            break

        neighbors.sort()
        best_neighbor_cost, best_neighbor, move = neighbors[0]
        current = best_neighbor

        if best_neighbor_cost < best_cost:
            best_cost = best_neighbor_cost
            best = best_neighbor[:]
            best_routes = decode(best, demands, cap, depot)

        tabu_list.append(move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return {
        "file": pathlib.Path(file_path).name,
        "algo": "Tabu Search (Basic)",
        "vehicles": len(best_routes),
        "distance": int(round(best_cost)),
        "time_sec": round(time.time() - start, 2),
        "routes": best_routes
    }

if __name__ == "__main__":
    import sys, json
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/cvrplib/A-n32-k5.vrp"
    result = solve(fp)
    print(json.dumps(result, indent=2))
