import math, random, time, os
from utils.cvrp_parser import read_vrp
from utils.constructive_ls import savings_initial

def savings_initial_solution(coords, demands, cap, depot):
    routes = savings_initial(cap, coords, demands, depot)
    chrom = [n for route in routes for n in route if n != depot]
    return chrom

def decode(chrom, demands, cap, depot):
    routes, load, r = [], 0, [depot]
    for n in chrom:
        if load + demands[n] <= cap:
            r.append(n); load += demands[n]
        else:
            r.append(depot); routes.append(r)
            r, load = [depot, n], demands[n]
    r.append(depot); routes.append(r)
    return routes

def route_length(route, coords):
    return sum(math.hypot(coords[route[i+1]][0] - coords[route[i]][0],
                          coords[route[i+1]][1] - coords[route[i]][1])
               for i in range(len(route)-1))

def cost(chrom, coords, demands, cap, depot):
    return sum(route_length(r, coords) for r in decode(chrom, demands, cap, depot))

def tournament_selection(pop, k=3):
    return min(random.sample(pop, k), key=lambda x: x[1])[0]

def solve(file_path, sec_limit=30, pop_size=150, mut_prob=0.2):
    cap, coords, demands = read_vrp(str(file_path))
    depot = 1
    nodes = [i for i in coords if i != depot]

    base = savings_initial_solution(coords, demands, cap, depot)
    population = [(base[:], cost(base, coords, demands, cap, depot))]
    for _ in range(pop_size - 1):
        perm = base[:]
        random.shuffle(perm)
        population.append((perm, cost(perm, coords, demands, cap, depot)))

    t0 = time.time()
    generation = 0
    while time.time() - t0 < sec_limit:
        generation += 1
        population.sort(key=lambda x: x[1])
        elite_cut = max(1, int(0.1 * pop_size))
        new_pop = population[:elite_cut]

        while len(new_pop) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            a, b = sorted(random.sample(range(len(p1)), 2))
            child = [None]*len(p1)
            child[a:b] = p1[a:b]
            fill = [x for x in p2 if x not in child[a:b]]
            ptr = 0
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = fill[ptr]
                    ptr += 1
            if random.random() < mut_prob:
                for _ in range(3):
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
            new_pop.append((child, cost(child, coords, demands, cap, depot)))

        population = new_pop

    best, best_cost = min(population, key=lambda x: x[1])
    routes = decode(best, demands, cap, depot)

    return {
        "file": os.path.basename(file_path),
        "algo": "Genetic GA Boosted",
        "vehicles": len(routes),
        "distance": int(round(best_cost)),
        "time_sec": round(time.time()-t0, 2),
        "routes": routes,
        "generations": generation
    }
