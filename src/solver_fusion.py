from __future__ import annotations
import random, math, time, json, sys, pathlib
from utils.cvrp_parser import read_vrp
from utils.constructive_ls import savings_initial, route_length
from utils.local_2opt import two_opt

# ---- Допоміжні функції ----
def total_len(routes, coords):
    return sum(route_length(r, coords) for r in routes)

def ruin(routes, k):
    ruined = []
    candidates = [r for r in routes if len(r) > 2]
    max_remove = sum(len(r) - 2 for r in candidates)
    k = min(k, max_remove)
    for _ in range(k):
        candidates = [r for r in routes if len(r) > 2]
        if not candidates:
            break
        r = random.choice(candidates)
        idx = random.randrange(1, len(r)-1)
        ruined.append(r.pop(idx))
        if len(r) == 2:
            routes.remove(r)
    return routes, ruined

def cheapest_insert(routes, client, cap, coords, demands, depot, max_positions=5):
    best_cost, best_r, best_pos = math.inf, None, None
    options = []
    for r in routes:
        load = sum(demands[v] for v in r)
        if load + demands[client] > cap:
            continue
        for pos in range(1, len(r)):
            extra = (route_length(r[pos-1:pos+1]+[r[pos]], coords) -
                     route_length(r[pos-1:pos+1], coords))
            options.append((extra, r, pos))
    options.sort()
    for extra, r, pos in options[:max_positions]:
        if extra < best_cost:
            best_cost, best_r, best_pos = extra, r, pos
    if best_r is None:
        routes.append([depot, client, depot])
    else:
        best_r.insert(best_pos, client)
    return routes

def recreate(routes, removed, cap, coords, demands, depot):
    for c in removed:
        routes = cheapest_insert(routes, c, cap, coords, demands, depot)
    return routes

def relocate(routes, coords, demands, cap, depot):
    improved = True
    while improved:
        improved = False
        for i, r1 in enumerate(routes):
            for j, r2 in enumerate(routes):
                if i == j or len(r1) <= 3:
                    continue
                for idx in range(1, len(r1)-1):
                    node = r1[idx]
                    load2 = sum(demands[v] for v in r2)
                    if load2 + demands[node] > cap:
                        continue
                    for pos in range(1, len(r2)):
                        new_r1 = r1[:idx] + r1[idx+1:]
                        new_r2 = r2[:pos] + [node] + r2[pos:]
                        old_len = route_length(r1, coords) + route_length(r2, coords)
                        new_len = route_length(new_r1, coords) + route_length(new_r2, coords)
                        if new_len < old_len - 1e-6:
                            r1[:] = new_r1
                            r2[:] = new_r2
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break
    routes[:] = [r for r in routes if len(r) > 2]
    return routes

def swap(routes, coords, demands, cap):
    improved = True
    while improved:
        improved = False
        for i, r1 in enumerate(routes):
            for j, r2 in enumerate(routes):
                if i >= j: continue
                for idx1 in range(1, len(r1)-1):
                    for idx2 in range(1, len(r2)-1):
                        n1, n2 = r1[idx1], r2[idx2]
                        load1 = sum(demands[v] for v in r1) - demands[n1] + demands[n2]
                        load2 = sum(demands[v] for v in r2) - demands[n2] + demands[n1]
                        if load1 > cap or load2 > cap:
                            continue
                        new_r1 = r1[:idx1] + [n2] + r1[idx1+1:]
                        new_r2 = r2[:idx2] + [n1] + r2[idx2+1:]
                        old_len = route_length(r1, coords) + route_length(r2, coords)
                        new_len = route_length(new_r1, coords) + route_length(new_r2, coords)
                        if new_len < old_len - 1e-6:
                            r1[:] = new_r1
                            r2[:] = new_r2
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break
    return routes

# ---- Основний алгоритм ----
def solve(file_path: str | pathlib.Path, sec_limit=30):
    cap, coords, demands = read_vrp(str(file_path))
    depot = 1
    n = len(demands) - 1

    # Savings + 2-opt як базовий старт
    routes = savings_initial(cap, coords, demands, depot)
    routes = [two_opt(r, coords, max_iter=8) for r in routes]
    best, best_len = [r.copy() for r in routes], total_len(routes, coords)

    # Вибір режиму
    if n <= 40:
        strategy = 'savings'
    elif n <= 80:
        strategy = 'light_rr'
    else:
        strategy = 'full_rr'

    print(f"Running strategy: {strategy}")

    start = time.time(); it = 0
    T = 0.2 * best_len

    while time.time() - start < sec_limit:
        it += 1
        if strategy == 'savings':
            break

        # Ruin & Recreate з локальним пошуком
        cur_routes = [r.copy() for r in routes]
        k_remove = 3 if strategy == 'light_rr' else random.randint(5, min(12, n//4))
        cur_routes, removed = ruin(cur_routes, k=k_remove)
        cur_routes = recreate(cur_routes, removed, cap, coords, demands, depot)
        cur_routes = [two_opt(r, coords, max_iter=4 if strategy == 'light_rr' else 8) for r in cur_routes]
        cur_routes = relocate(cur_routes, coords, demands, cap, depot)
        cur_routes = swap(cur_routes, coords, demands, cap)

        cur_len = total_len(cur_routes, coords)
        delta = cur_len - best_len
        if delta < 0 or random.random() < math.exp(-delta / T):
            routes = [r.copy() for r in cur_routes]
            if cur_len < best_len:
                best, best_len = [r.copy() for r in cur_routes], cur_len

        if it % 10 == 0:
            print(f"[{it}] Best: {best_len:.2f}  Temp: {T:.3f}  Routes: {len(routes)}")
        if it % 50 == 0:
            T *= 0.95

    return {
        "file": pathlib.Path(file_path).name,
        "algo": f"Auto-{strategy}",
        "vehicles": len(best),
        "distance": int(best_len),
        "time_sec": round(time.time() - start, 2),
        "routes": best,
    }

# ---- Запуск як скрипта ----
if __name__ == "__main__":
    fp  = sys.argv[1] if len(sys.argv) > 1 else "data/cvrplib/A-n32-k5.vrp"
    sec = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(json.dumps(solve(fp, sec), indent=2))
