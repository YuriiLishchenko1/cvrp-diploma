#solver_tabu.py
# import time
# import random
# import pathlib
# from utils.cvrp_parser import read_vrp
# from utils.constructive_ls import savings_initial, route_length

# def solve(file_path, sec_limit=30, tabu_tenure=10):
    # """
    # Tabu Search for CVRP with initial solution from Clarke–Wright + 2-opt moves.
    # """
    # cap, coords, demands = read_vrp(str(file_path))
    # depot = 1

    #1) Початкове рішення: Savings (Clarke–Wright)
    # init_routes = savings_initial(cap, coords, demands, depot)
    #«flat» послідовність клієнтів без депо
    # flat = [n for route in init_routes for n in route if n != depot]

    #функція декодування flat -> список маршрутів за вантажопідйомністю
    # def decode(chrom):
        # routes, load, r = [], 0, [depot]
        # for n in chrom:
            # if load + demands[n] <= cap:
                # r.append(n)
                # load += demands[n]
            # else:
                # r.append(depot)
                # routes.append(r)
                # r, load = [depot, n], demands[n]
        # r.append(depot)
        # routes.append(r)
        # return routes

    #оцінка вартості розв'язку
    # def cost(chrom):
        # return sum(route_length(r, coords) for r in decode(chrom))

    #2) Налаштування Tabu
    # t0 = time.time()
    # best_flat = flat[:]
    # best_cost = cost(best_flat)
    # curr_flat = best_flat[:]
    # curr_cost = best_cost

    # tabu_list = []
    # tenure = tabu_tenure

    #3) Tabu-пошук
    # iteration = 0
    # while time.time() - t0 < sec_limit:
        # iteration += 1
        # best_neighbor = None
        # best_neighbor_cost = float('inf')
        # best_move = None

        #генеруємо всі 2-opt сусідства (swap i,j)
        # n = len(curr_flat)
        # for i in range(n-1):
            # for j in range(i+1, n):
                # move = (i, j)
                # if move in tabu_list:
                    # continue
                #створити новий flat
                # cand = curr_flat[:]
                # cand[i], cand[j] = cand[j], cand[i]
                # c = cost(cand)
                # if c < best_neighbor_cost:
                    # best_neighbor_cost = c
                    # best_neighbor = cand
                    # best_move = move

        # if best_neighbor is None:
            # break

        #оновити поточне
        # curr_flat = best_neighbor
        # curr_cost = best_neighbor_cost

        #порівняти з глобальним
        # if curr_cost < best_cost:
            # best_flat = curr_flat[:]
            # best_cost = curr_cost

        #оновити Tabu-list
        # tabu_list.append(best_move)
        # if len(tabu_list) > tenure:
            # tabu_list.pop(0)

    #фінальний розв'язок
    # best_routes = decode(best_flat)
    # total_time = time.time() - t0

    # return {
        # "file": pathlib.Path(file_path).name,
        # "algo": "Tabu Search",
        # "vehicles": len(best_routes),
        # "distance": best_cost,
        # "time_sec": round(total_time, 2),
        # "routes": best_routes
    # }


# if __name__ == "__main__":
    # import sys, json
    # fp = sys.argv[1] if len(sys.argv)>1 else "data/cvrplib/A-n32-k5.vrp"
    # print(json.dumps(solve(fp), indent=2))


# solver_tabu.py
import time
import random
import pathlib
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
    diversification_rate=0.1
):
    """
    Tabu Search for CVRP with enhancements:
    - time and iteration limits
    - no-improvement diversification
    - candidate list size
    - explicit aspiration rule
    """
    data = read_vrp(file_path)
    cap     = data["capacity"]
    coords  = data["coords"]
    demands = data["demand"]
    depot   = data["depot_id"]
    depot = 1

    # 1) Initial solution: Clarke–Wright Savings
    init_routes = savings_initial(cap, coords, demands, depot)
    flat = [n for route in init_routes for n in route if n != depot]

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

    # settings
    t0 = time.time()
    best_flat = flat[:]
    best_cost = cost(best_flat)
    curr_flat = best_flat[:]
    curr_cost = best_cost

    tabu_list = []
    iterations = 0
    no_improve_count = 0

    # main loop
    while True:
        # time-based stopping
        if time.time() - t0 >= sec_limit:
            break
        # iteration-based stopping
        if max_iters and iterations >= max_iters:
            break
        if st.session_state.get("stop_flag"):  # ⛔ якщо користувач натиснув "Зупинити"
            break

        iterations += 1
        # generate moves
        moves = [(i, j) for i in range(len(curr_flat)-1) for j in range(i+1, len(curr_flat))]
        # limit candidates
        if candidate_list_size and candidate_list_size < len(moves):
            moves = random.sample(moves, candidate_list_size)

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

        # update global best
        if curr_cost < best_cost:
            best_flat, best_cost = curr_flat[:], curr_cost
            no_improve_count = 0
        else:
            no_improve_count += 1

        # update tabu list
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # diversification
        if no_improve_count >= no_improve_limit:
            k = max(1, int(len(curr_flat) * diversification_rate))
            for _ in range(k):
                i, j = random.sample(range(len(curr_flat)), 2)
                curr_flat[i], curr_flat[j] = curr_flat[j], curr_flat[i]
            curr_cost = cost(curr_flat)
            no_improve_count = 0

    best_routes = decode(best_flat)
    total_time = time.time() - t0

    return {
        "file": pathlib.Path(file_path).name,
        "algo": "Tabu Search",
        "vehicles": len(best_routes),
        "distance": best_cost,
        "time_sec": round(total_time, 2),
        "routes": best_routes,
        "iterations": iterations
    }

if __name__ == "__main__":
    import sys, json
    fp = sys.argv[1] if len(sys.argv)>1 else "data/cvrplib/A-n32-k5.vrp"
    print(json.dumps(solve(fp), indent=2))

