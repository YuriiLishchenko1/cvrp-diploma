import random, time, math, numpy as np
from utils.cvrp_parser import read_vrp

def distance(p1, p2):
    return int(math.hypot(p1[0]-p2[0], p1[1]-p2[1]))

def solve_ga(file_path, sec_limit=120, pop_size=150):
    cap, coords, demands = read_vrp(file_path)
    nodes = list(coords.keys())[1:]          # без depot=1
    dist = {(i, j): distance(coords[i], coords[j]) for i in coords for j in coords}
    
    gen = 0

    def split_by_capacity(chrom):
        route, load, routes = [1], 0, []
        for n in chrom:
            if load + demands[n] <= cap:
                route.append(n); load += demands[n]
            else:
                route.append(1); routes.append(route)
                route, load = [1, n], demands[n]
        route.append(1); routes.append(route)
        return routes

    def route_cost(rt):
        return sum(dist[(rt[i], rt[i+1])] for i in range(len(rt)-1))

    def two_opt(route):
        best = route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j-i == 1: continue
                    new = best[:i] + best[i:j][::-1] + best[j:]
                    if route_cost(new) < route_cost(best):
                        best, improved = new, True
        return best

    def tournament(pop, k=3):
        return min(random.sample(pop, k), key=fitness)

    def fitness(chrom):
        rts = split_by_capacity(chrom)
        return sum(route_cost(r) for r in rts)

    # ------- GA -------
    pop = [random.sample(nodes, len(nodes)) for _ in range(pop_size)]
    best = min(pop, key=fitness)
    t0 = time.time()
    
    ELITE = 5                     # <– 1. elitism
    
    while time.time() - t0 < sec_limit:
        # ----------- селекція -----------
        pop.sort(key=fitness)              # від найкращого до гіршого
        elites  = pop[:ELITE]              # елітні особини
        parents = [tournament(pop) for _ in range(pop_size//2)]

        # ----------- кросовер + мутація -----------
        children = []
        while len(children) < pop_size//2:
            p1, p2 = random.sample(parents, 2)
            cut = random.randint(1, len(nodes) - 2)
            child = p1[:cut] + [g for g in p2 if g not in p1[:cut]]

            # мутація
            if random.random() < 0.2:
                i, j = random.sample(range(len(nodes)), 2)
                child[i], child[j] = child[j], child[i]

            children.append(child)

        new_pop = children + elites        # ← готова нова популяція
        pop = new_pop

        # ----------- відслідковуємо прогрес -----------
        cand = min(pop, key=fitness)
        if fitness(cand) < fitness(best):
            best = cand

        if gen % 50 == 0:
            print(f"gen={gen:4}  best_dist={fitness(best):.1f}")

        gen += 1



    best_routes = [two_opt(r) for r in split_by_capacity(best)]
    total = sum(route_cost(r) for r in best_routes)
    return {
        "file": file_path.split("\\")[-1],
        "vehicles": len(best_routes),
        "distance": total,
        "time_sec": round(time.time() - t0, 2),
        "routes": best_routes,
    }
    

if __name__ == "__main__":
    import sys, json, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--time", type=int, default=60)
    ap.add_argument("--pop",  type=int, default=100)
    args = ap.parse_args()

    res = solve_ga(args.file, args.time, args.pop)
    print(json.dumps(res, indent=2))

