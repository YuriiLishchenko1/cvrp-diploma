# solver_tabu_boost.py
import time
import pathlib
import numpy as np
from utils.cvrp_parser import read_vrp
from utils.constructive_ls import savings_initial, route_length
import streamlit as st


def solve(
    file_path,
    sec_limit=30,
    tabu_tenure=10,
    max_iters=None,
    no_improve_limit=50,
    candidate_list_size=None,
    aspiration_threshold=0,
    diversification_rate=0.1,
    n_restarts=3,
    seed_base=42
):
    data = read_vrp(file_path)
    cap     = data["capacity"]
    coords  = data["coords"]
    demands = data["demand"]
    depot   = data["depot_id"]
    depot = 1

    def decode(chrom):
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

    def cost(chrom):
        return sum(route_length(r, coords) for r in decode(chrom))

    best_overall_flat = None
    best_overall_cost = float('inf')
    best_routes_final = None
    total_time = 0
    total_iterations = 0

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed_base + restart)

        init_routes = savings_initial(cap, coords, demands, depot)
        flat = [n for route in init_routes for n in route if n != depot]

        t0 = time.time()
        curr_flat = flat[:]
        rng.shuffle(curr_flat)
        curr_cost = cost(curr_flat)
        best_flat = curr_flat[:]
        best_cost = curr_cost

        tabu_list = []
        iterations = 0
        no_improve_count = 0

        while True:
            if time.time() - t0 >= sec_limit:
                break
            if max_iters and iterations >= max_iters:
                break
            if st.session_state.get("stop_flag"):
                break

            iterations += 1
            moves = [(i, j) for i in range(len(curr_flat)-1) for j in range(i+1, len(curr_flat))]
            if candidate_list_size and candidate_list_size < len(moves):
                moves = rng.choice(moves, size=candidate_list_size, replace=False).tolist()

            best_neighbor, best_move = None, None
            best_neighbor_cost = float('inf')

            for move in moves:
                if move in tabu_list and (curr_cost - best_cost) < aspiration_threshold:
                    continue
                i, j = move
                cand = curr_flat[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = cost(cand)
                if c < best_neighbor_cost:
                    best_neighbor_cost = c
                    best_neighbor = cand
                    best_move = move

            if not best_neighbor:
                break

            curr_flat, curr_cost = best_neighbor, best_neighbor_cost

            if curr_cost < best_cost:
                best_flat, best_cost = curr_flat[:], curr_cost
                no_improve_count = 0
            else:
                no_improve_count += 1

            tabu_list.append(best_move)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

            if no_improve_count >= no_improve_limit:
                k = max(1, int(len(curr_flat) * diversification_rate))
                for _ in range(k):
                    i, j = rng.integers(0, len(curr_flat), size=2)
                    curr_flat[i], curr_flat[j] = curr_flat[j], curr_flat[i]
                curr_cost = cost(curr_flat)
                no_improve_count = 0

        if best_cost < best_overall_cost:
            best_overall_flat = best_flat[:]
            best_overall_cost = best_cost
            best_routes_final = decode(best_flat)

        total_time += time.time() - t0
        total_iterations += iterations

    return {
        "file": pathlib.Path(file_path).name,
        "algo": "Tabu Boosted",
        "vehicles": len(best_routes_final),
        "distance": round(best_overall_cost, 2),
        "time_sec": round(total_time, 2),
        "routes": best_routes_final,
        "iterations": total_iterations
    }


if __name__ == "__main__":
    import sys, json
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/cvrplib/A-n32-k5.vrp"
    print(json.dumps(solve(fp), indent=2))
