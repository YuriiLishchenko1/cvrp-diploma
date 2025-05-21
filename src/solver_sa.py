import random, time, math, pathlib, json, sys
from utils.cvrp_parser import read_vrp
from utils.local_2opt import two_opt
from utils.constructive_ls import route_length

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

def solve(file_path, sec_limit=30, T0=500, cool=0.995):
    cap, coords, demands = read_vrp(str(file_path))
    nodes = [i for i in coords if i != 1]
    depot = 1

    # початкове рішення (випадкове)
    best = curr = nodes[:]
    random.shuffle(curr)
    best_f = curr_f = cost(curr, coords, cap, demands, depot)

    t0 = time.time(); T = T0
    while time.time()-t0 < sec_limit:
        i,j = random.sample(range(len(nodes)),2)
        cand = curr[:]
        cand[i], cand[j] = cand[j], cand[i]
        cand_f = cost(cand, coords, cap, demands, depot)
        if cand_f < curr_f or random.random() < math.exp(-(cand_f-curr_f)/T):
            curr, curr_f = cand, cand_f
            if cand_f < best_f:
                best, best_f = cand, cand_f
        T *= cool

    # локальний 2-opt на кінцевих маршрутах
    routes = [_ for _ in _decode(best, cap, demands, depot)]
    routes = [two_opt(r, coords, first_improve=False) for r in routes]
    best_f = sum(route_length(r, coords) for r in routes)

    return {
        "file": pathlib.Path(file_path).name,
        "algo": "SimAnn-2opt",
        "vehicles": len(routes),
        "distance": round(best_f, 2),
        "time_sec": round(time.time()-t0, 2),
        "routes": routes,
    }

if __name__=="__main__":
    fp = sys.argv[1] if len(sys.argv)>1 else "data/cvrplib/A-n32-k5.vrp"
    print(json.dumps(solve(fp), indent=2))
