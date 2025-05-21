"""
Ant-Colony optimiser для задачі CVRP.

▪ повертає такий самий json-словник, як усі інші solver-и
▪ працює тільки з евклідовими задачами з cvrplib
▪ запускається окремо:
      python -m src.solver_aco data/cvrplib/A-n32-k5.vrp 30
"""
from __future__ import annotations
import time, math, random, json, sys, pathlib
from collections import defaultdict
from typing import List

from utils.cvrp_parser import read_vrp
from utils.constructive_ls import route_length
from utils.local_2opt import two_opt   # існуючий 2-opt

def solve(
    file_path: str | pathlib.Path,
    sec_limit: int = 30,
    n_ants: int = 25,
    alpha: float = 1.0,     # вплив феромону
    beta : float = 4.0,     # вплив евристики (1/дистанція)
    rho  : float = 0.1,     # швидкість випаровування
):
    # 1) зчитаємо дані
    cap, coords, demands = read_vrp(str(file_path))
    depot = 1
    # усі клієнти, крім депо
    nodes = [i for i in coords if i != depot]

    # 2) матриця відстаней
    dist = {
      (i, j): int(math.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1]))
      for i in coords for j in coords
    }

    # 3) початкові феромони та евристика
    tau0 = 1 / (len(nodes) * (sum(dist.values())/len(dist)))
    pher = defaultdict(lambda: tau0)
    eta  = {
      (i, j): 1 / max(dist[(i, j)], 1)
      for i in coords for j in coords if i != j
    }

    best_routes, best_len = None, float("inf")
    t_start = time.time()

    # головний цикл
    while time.time() - t_start < sec_limit:
        all_solutions = []

        for ant in range(n_ants):
            routes = []
            unvisited = set(nodes)

            # генеруємо маршрути для однієї "мурашки"
            while unvisited:
                load, curr, route = 0, depot, [depot]

                # ростемо, доки є кому завантажувати
                while True:
                    feas = [
                      j for j in unvisited
                      if load + demands[j] <= cap
                    ]
                    if not feas:
                        break

                    # roulette-wheel selection
                    weights = []
                    total = 0.0
                    for j in feas:
                        w = (pher[(curr, j)] ** alpha) * (eta[(curr, j)] ** beta)
                        weights.append((j, w))
                        total += w
                    pick = random.random() * total
                    cum = 0.0
                    for j, w in weights:
                        cum += w
                        if cum >= pick:
                            nxt = j
                            break

                    route.append(nxt)
                    unvisited.remove(nxt)
                    load += demands[nxt]
                    curr = nxt

                # завершуємо маршрут
                route.append(depot)
                # локальний 2-opt
                route = two_opt(route, coords)
                routes.append(route)

            # довжина цього рішення
            tot_len = sum(route_length(r, coords) for r in routes)
            all_solutions.append((routes, tot_len))

            if tot_len < best_len:
                best_len, best_routes = tot_len, routes

        # випаровування
        for key in list(pher.keys()):
            pher[key] *= (1 - rho)

        # відкладаємо від найкращої з цього покоління
        gen_best_routes, gen_best_len = min(all_solutions, key=lambda x: x[1])
        for r in gen_best_routes:
            for a, b in zip(r[:-1], r[1:]):
                pher[(a, b)] += rho * (1 / gen_best_len)

    return {
        "file": pathlib.Path(file_path).name,
        "algo": "Ant Colony",
        "vehicles": len(best_routes),
        "distance": int(best_len),
        "time_sec": round(time.time() - t_start, 2),
        "routes": best_routes,
    }

if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/cvrplib/A-n32-k5.vrp"
    sec = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(json.dumps(solve(fp, sec), indent=2))
