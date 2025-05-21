#!/usr/bin/env python3
import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from threading import Thread

# ==== PARSING CVRP INSTANCES ====
def parse_vrp_file(filepath):
    coords, demands = {}, {}
    capacity = None
    max_vehicles = None
    optimal = None
    depot = None
    with open(filepath) as f:
        lines = f.readlines()
    section = None
    for line in lines:
        tok = line.strip().split()
        if not tok: continue
        key = tok[0]
        if key == 'CAPACITY':
            capacity = int(tok[-1])
            continue
        if key == 'COMMENT':
            if 'No of trucks:' in line:
                max_vehicles = int(line.split('No of trucks:')[1].split(',')[0])
            if 'Optimal value:' in line:
                optimal = int(line.split('Optimal value:')[1].split(')')[0])
            continue
        if key == 'NODE_COORD_SECTION': section = 'NODE'; continue
        if key == 'DEMAND_SECTION': section = 'DEMAND'; continue
        if key == 'DEPOT_SECTION': section = 'DEPOT'; continue
        if key == 'EOF': break
        if section == 'NODE':
            idx = int(tok[0]); coords[idx] = (float(tok[1]), float(tok[2])); continue
        if section == 'DEMAND':
            idx = int(tok[0]); demands[idx] = int(tok[1]); continue
        if section == 'DEPOT':
            idx = int(tok[0])
            if idx != -1: depot = coords[idx]
    customers = []
    for idx in sorted(demands):
        if demands[idx] > 0:
            customers.append({'coords': coords[idx], 'demand': demands[idx]})
    return customers, depot, capacity, max_vehicles, optimal

# ==== UTILITIES ====

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def build_matrix(customers, depot):
    pts = [depot] + [c['coords'] for c in customers]
    n = len(pts)
    M = [[distance(pts[i], pts[j]) for j in range(n)] for i in range(n)]
    return np.array(M)

def total_cost(routes, M):
    cost = 0
    for r in routes:
        prev = 0
        for c in r:
            cost += M[prev][c+1]
            prev = c+1
        cost += M[prev][0]
    return cost

# Check capacity feasibility
def is_feasible(routes, customers, capacity):
    for r in routes:
        load = sum(customers[i]['demand'] for i in r)
        if load > capacity:
            return False
    return True

# ==== ALGORITHMS ====
# Savings + 2-Opt
def solve_savings_2opt(customers, depot, capacity, M, tlim):
    n = len(customers)
    # construct savings list
    savings = [(i, j, M[0][i+1] + M[0][j+1] - M[i+1][j+1])
               for i in range(n) for j in range(i+1, n)]
    savings.sort(key=lambda x: x[2], reverse=True)
    routes = [[i] for i in range(n)]
    # merge routes greedily
    for i, j, _ in savings:
        ri = next(r for r in routes if i in r)
        rj = next(r for r in routes if j in r)
        if ri is not rj and sum(customers[k]['demand'] for k in ri + rj) <= capacity:
            ri.extend(rj)
            routes.remove(rj)
    # local 2-opt
    for idx, r in enumerate(routes):
        best = r[:]
        improved = True
        while improved:
            improved = False
            for a in range(1, len(best)-1):
                for b in range(a+1, len(best)):
                    if b - a < 2: continue
                    cand = best[:a] + best[a:b][::-1] + best[b:]
                    if total_cost([cand], M) < total_cost([best], M):
                        best, improved = cand, True
            r = best
        routes[idx] = best
    return routes

# Random neighbor with capacity check
def get_random_neighbor(routes, customers, capacity):
    for _ in range(100):
        new = [r[:] for r in routes if r]
        if not new:
            return new
        i = random.randrange(len(new)); r1 = new[i]
        if not r1: continue
        c = random.choice(r1)
        r1.remove(c)
        j = random.randrange(len(new));
        pos = random.randrange(len(new[j]) + 1)
        new[j].insert(pos, c)
        new = [r for r in new if r]
        if is_feasible(new, customers, capacity):
            return new
    return routes

# Simulated Annealing
def solve_simulated_annealing(customers, depot, capacity, M, tlim):
    start = time.time()
    curr = solve_savings_2opt(customers, depot, capacity, M, tlim)
    best = curr[:]
    bc = total_cost(best, M)
    T, alpha = 1000, 0.995
    while time.time() - start < tlim:
        nei = get_random_neighbor(curr, customers, capacity)
        nc = total_cost(nei, M)
        cc = total_cost(curr, M)
        if nc < cc or random.random() < math.exp((cc - nc)/T):
            curr, cc = nei, nc
        if nc < bc:
            best, bc = nei, nc
        T *= alpha
    return best

# Tabu Search
def solve_tabu_search(customers, depot, capacity, M, tlim):
    start = time.time()
    curr = solve_savings_2opt(customers, depot, capacity, M, tlim)
    best = curr[:]
    bc = total_cost(best, M)
    tabu = deque(maxlen=100)
    while time.time() - start < tlim:
        neighs = [get_random_neighbor(curr, customers, capacity) for _ in range(50)]
        neighs.sort(key=lambda r: total_cost(r, M))
        for n in neighs:
            key = tuple(tuple(r) for r in n)
            if key not in tabu:
                curr = n; tabu.append(key)
                break
        cc = total_cost(curr, M)
        if cc < bc:
            best, bc = curr, cc
    return best

# ALNS
def solve_alns(customers, depot, capacity, M, tlim):
    start = time.time()
    routes = solve_savings_2opt(customers, depot, capacity, M, tlim)
    best = routes[:]
    bc = total_cost(best, M)
    while time.time() - start < tlim:
        allc = [c for r in routes for c in r]
        k = max(1, int(len(allc) * 0.2))
        removed = random.sample(allc, k)
        part = [[c for c in r if c not in removed] for r in routes]
        for c in removed:
            bi = None; best_inc = float('inf')
            for i, r in enumerate(part):
                load = sum(customers[v]['demand'] for v in r)
                if load + customers[c]['demand'] > capacity: continue
                for pos in range(len(r)+1):
                    prev = r[pos-1] if pos>0 else -1
                    nxt = r[pos] if pos<len(r) else -1
                    inc = (M[prev+1][c+1] if prev>=0 else M[0][c+1]) + (M[c+1][nxt+1] if nxt>=0 else M[c+1][0]) - (M[prev+1][nxt+1] if prev>=0 or nxt>=0 else 0)
                    if inc < best_inc:
                        best_inc, bi = inc, (i, pos)
            if bi: part[bi[0]].insert(bi[1], c)
            else: part.append([c])
        routes = part
        cst = total_cost(routes, M)
        if cst < bc:
            best, bc = routes[:], cst
    return best

# Genetic Algorithm
def solve_genetic(customers, depot, capacity, M, tlim):
    start = time.time()
    n = len(customers)
    pop = [random.sample(range(n), n) for _ in range(20)]
    best_perm = min(pop, key=lambda p: total_cost(decode_perm(p, customers, capacity), M))
    bc = total_cost(decode_perm(best_perm, customers, capacity), M)
    while time.time() - start < tlim:
        pop.sort(key=lambda p: total_cost(decode_perm(p, customers, capacity), M))
        next_pop = pop[:5]
        while len(next_pop) < len(pop):
            p1, p2 = random.sample(pop[:10], 2)
            cut = random.randint(1, n-1)
            child = p1[:cut] + [c for c in p2 if c not in p1[:cut]]
            if random.random() < 0.2:
                i, j = random.sample(range(n), 2); child[i], child[j] = child[j], child[i]
            next_pop.append(child)
        pop = next_pop
        curr = pop[0]
        cc = total_cost(decode_perm(curr, customers, capacity), M)
        if cc < bc:
            best_perm, bc = curr, cc
    return decode_perm(best_perm, customers, capacity)

def decode_perm(perm, customers, capacity):
    routes, cur, load = [], [], 0
    for c in perm:
        d = customers[c]['demand']
        if load + d <= capacity: cur.append(c); load += d
        else: routes.append(cur); cur, load = [c], d
    if cur: routes.append(cur)
    return routes

# Ant Colony Optimization
def solve_aco(customers, depot, capacity, M, tlim):
    start = time.time()
    n = len(customers); ants = n
    pher = np.ones((n+1, n+1)); alpha, beta, rho = 1.0, 2.0, 0.1
    best, bc = None, float('inf')
    while time.time() - start < tlim:
        all_sols = []
        for _ in range(ants):
            un = set(range(n)); routes = []
            while un:
                cur, load, r = -1, 0, []
                while True:
                    choices = []
                    for j in list(un):
                        if load + customers[j]['demand'] <= capacity:
                            tau = pher[cur+1][j+1]; eta = 1.0/(M[cur+1][j+1]+1e-6)
                            choices.append((j, tau**alpha * eta**beta))
                    if not choices: break
                    tot = sum(w for _, w in choices); pick = random.random()*tot; cum = 0
                    for j, w in choices:
                        cum += w
                        if cum >= pick: nxt = j; break
                    un.remove(nxt); r.append(nxt); load += customers[nxt]['demand']; cur = nxt
                routes.append(r)
            cst = total_cost(routes, M)
            if cst < bc: best, bc = routes, cst
            all_sols.append((routes, cst))
        pher *= (1-rho)
        for sol, cst in all_sols:
            for r in sol:
                prev = 0
                for v in r:
                    pher[prev][v+1] += 1.0/cst; pher[v+1][prev] += 1.0/cst; prev = v+1
                pher[prev][0] += 1.0/cst
    return best

# Hybrid ALNS + Tabu

def solve_hybrid(customers, depot, capacity, M, tlim):
    start = time.time()
    t1 = tlim * 0.7
    routes = solve_alns(customers, depot, capacity, M, t1)
    best = routes[:]; bc = total_cost(best, M)
    tabu = deque(maxlen=100)
    while time.time() - start < tlim:
        neigh = get_random_neighbor(routes, customers, capacity)
        key = tuple(tuple(r) for r in neigh)
        if key in tabu: continue
        tabu.append(key); routes = neigh
        cst = total_cost(routes, M)
        if cst < bc: best, bc = neigh, cst
    return best

# ==== DRIVER + PLOTTING ====
def compare_and_plot(filepath, time_limit, output_widget):
    try:
        customers, depot, cap, max_v, opt = parse_vrp_file(filepath)
        M = build_matrix(customers, depot)
        algos = {
            'Savings+2Opt': solve_savings_2opt,
            'SimAnneal': solve_simulated_annealing,
            'Tabu': solve_tabu_search,
            'ALNS': solve_alns,
            'Genetic': solve_genetic,
            'ACO': solve_aco,
            'Hybrid': solve_hybrid
        }
        results = {}
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, f"Results for {filepath}\nTime limit: {time_limit}s\n")
        output_widget.insert(tk.END, f"{'Alg':<12}{'Dist':<10}{'Time':<8}{'Veh':<5}{'Dev%':<7}\n")
        for name, func in algos.items():
            t0 = time.time(); routes = func(customers, depot, cap, M, time_limit)
            dt = time.time() - t0; d = total_cost(routes, M)
            dev = (d - opt)/opt*100 if opt else 0; v = len(routes)
            output_widget.insert(tk.END, f"{name:<12}{d:<10.2f}{dt:<8.2f}{v:<5}{dev:<7.2f}\n")
            results[name] = (routes, d)
        for name, (routes, dist) in results.items():
            plt.figure(figsize=(6,4))
            loads = [sum(customers[i]['demand'] for i in r) for r in routes]
            for idx, route in enumerate(routes):
                x = [depot[0]] + [customers[i]['coords'][0] for i in route] + [depot[0]]
                y = [depot[1]] + [customers[i]['coords'][1] for i in route] + [depot[1]]
                plt.plot(x, y, marker='o', label=f"Route {idx+1} (Load: {loads[idx]}/{cap})")
            plt.scatter(depot[0], depot[1], c='red', s=100, label='Depot')
            plt.title(f"{name}: {dist:.2f}")
            plt.legend(loc='upper right', fontsize='small')
            plt.tight_layout(); plt.show()
    except Exception as e:
        messagebox.showerror('Error', str(e))

def launch_gui():
    root = tk.Tk(); root.title('CVRP Solver Comparison')
    frm = tk.Frame(root); frm.pack(padx=10, pady=10)
    btn_load = tk.Button(frm, text='Load CVRP File', width=20)
    btn_load.grid(row=0, column=0, padx=5, pady=5)
    lbl_time = tk.Label(frm, text='Time limit (s):'); lbl_time.grid(row=0, column=1)
    ent_time = tk.Entry(frm, width=5); ent_time.insert(0, '60'); ent_time.grid(row=0, column=2)
    txt_output = scrolledtext.ScrolledText(root, width=80, height=20); txt_output.pack(padx=10, pady=10)
    def on_load():
        file = filedialog.askopenfilename(filetypes=[('VRP files', '*.vrp;*.txt')])
        if not file: return
        try: tlim = int(ent_time.get())
        except: messagebox.showerror('Error', 'Invalid time limit'); return
        Thread(target=compare_and_plot, args=(file, tlim, txt_output), daemon=True).start()
    btn_load.config(command=on_load)
    root.mainloop()

if __name__ == '__main__':
    launch_gui()
