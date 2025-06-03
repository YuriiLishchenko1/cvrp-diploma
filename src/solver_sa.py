import random, time, math, pathlib, json, sys
from utils.cvrp_parser import read_vrp
from utils.constructive_ls import route_length, savings_initial
import streamlit as st

def _decode(chrom, cap, demands, depot):
    routes, load, r = [], 0, [depot]
    for n in chrom:
        if load+demands[n] <= cap:
            r.append(n); load += demands[n]
        else:
            r.append(depot); routes.append(r)
            r, load = [depot, n], demands[n]
    r.append(depot); routes.append(r)
    return routes

def cost(chrom, coords, cap, demands, depot):
    return sum(route_length(r, coords) for r in _decode(chrom, cap, demands, depot))

def solve(file_path, sec_limit=30, T0=None, cool=0.995):
    data = read_vrp(file_path)
    cap     = data["capacity"]
    coords  = data["coords"]
    demands = data["demand"]
    depot   = data["depot_id"]
    depot = 1
    nodes = [i for i in coords if i != depot]

    # стартове рішення: Savings
    initial = [n for r in savings_initial(cap, coords, demands, depot) for n in r if n != depot]
    curr = best = initial[:]

    best_f = curr_f = cost(curr, coords, cap, demands, depot)

    T0 = T0 or (50 * len(nodes))
    t0 = time.time()
    T = T0

    while time.time()-t0 < sec_limit:
        if st.session_state.get("stop_flag"):
            break

        i, j = random.sample(range(len(nodes)), 2)
        cand = curr[:]
        cand[i], cand[j] = cand[j], cand[i]
        cand_f = cost(cand, coords, cap, demands, depot)

        if cand_f < curr_f or random.random() < math.exp(-(cand_f - curr_f) / T):
            curr, curr_f = cand, cand_f
            if cand_f < best_f:
                best, best_f = cand, cand_f

        T = T0 * (1 - (time.time() - t0) / sec_limit)

    routes = _decode(best, cap, demands, depot)

    return {
        "file": pathlib.Path(file_path).name,
        "algo": "SimAnn",
        "vehicles": len(routes),
        "distance": round(best_f, 2),
        "time_sec": round(time.time() - t0, 2),
        "routes": routes,
    }

if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/cvrplib/A-n32-k5.vrp"
    print(json.dumps(solve(fp), indent=2))
