import random, time, math, pathlib
from utils.cvrp_parser import read_vrp
from utils.constructive_ls import route_length

def solve(file_path, sec_limit=30, pop_size=150, mut_prob=0.2, elit_frac=0.1):
    cap, coords, demands = read_vrp(str(file_path))
    depot = 1
    nodes = [i for i in coords if i != depot]

    # стартове рішення: Savings
    from utils.constructive_ls import savings_initial
    init_routes = savings_initial(cap, coords, demands, depot)
    best_flat = [n for r in init_routes for n in r if n != depot]

    def decode(chrom):
        routes, load, r = [], 0, [depot]
        for n in chrom:
            if load + demands[n] <= cap:
                r.append(n); load += demands[n]
            else:
                r.append(depot); routes.append(r)
                r, load = [depot, n], demands[n]
        r.append(depot); routes.append(r)
        return routes

    def cost(chrom):
        return sum(route_length(r, coords) for r in decode(chrom))

    # генетика
    population = []
    for _ in range(pop_size):
        perm = best_flat[:]
        random.shuffle(perm)
        population.append((perm, cost(perm)))

    t0 = time.time()
    while time.time() - t0 < sec_limit:
        # відбір еліти
        population.sort(key=lambda x: x[1])
        elite_cut = int(elit_frac * pop_size)
        new_pop = population[:elite_cut]

        # кросовер + мутація
        while len(new_pop) < pop_size:
            p1, _ = random.choice(population[:pop_size//2])
            p2, _ = random.choice(population[:pop_size//2])
            # two-point crossover
            a, b = sorted(random.sample(range(len(p1)), 2))
            child = [None]*len(p1)
            child[a:b] = p1[a:b]
            fill = [x for x in p2 if x not in child[a:b]]
            ptr = 0
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = fill[ptr]
                    ptr += 1
            # мутація
            if random.random() < mut_prob:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]

            new_pop.append((child, cost(child)))

        population = new_pop

    # після таймауту найкращий
    best_perm, best_cost = min(population, key=lambda x: x[1])
    routes = decode(best_perm)

    return {
        "file": pathlib.Path(file_path).name,
        "algo": "Genetic GA",
        "vehicles": len(routes),
        "distance": best_cost,
        "time_sec": round(time.time()-t0, 2),
        "routes": routes
    }

if __name__=="__main__":
    import sys, json
    fp = sys.argv[1] if len(sys.argv)>1 else "data/cvrplib/A-n32-k5.vrp"
    print(json.dumps(solve(fp), indent=2))
