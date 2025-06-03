# файл: src/utils/constructive_ls.py

from utils.cvrp_parser import read_vrp
from math import hypot
from collections import defaultdict
import math
import streamlit as st

def eucl(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])

def route_length(route, coords):
    dist = 0
    for i,j in zip(route[:-1], route[1:]):
        dist += eucl(coords[i], coords[j])
    return dist

def savings_initial(cap, coords, demands, depot=1, max_vehicles=None):
    # 1. Початкові маршрути: кожен клієнт окремо
    routes = {i: [depot, i, depot] for i in coords if i != depot}
    loads  = {i: demands[i] for i in coords if i != depot}

    # 2. Обчислюємо savings
    savings = []
    for i in coords:
        for j in coords:
            if i < j and i != depot and j != depot:
                d_ij = eucl(coords[i], coords[j])
                d_id = eucl(coords[i], coords[depot])
                d_jd = eucl(coords[j], coords[depot])
                savings.append((i, j, d_id + d_jd - d_ij))
    savings.sort(key=lambda x: x[2], reverse=True)

    # 3. Головне злиття
    for i, j, s in savings:
        if max_vehicles and len(routes) <= max_vehicles:
            break  # ✋ Зупиняємо, коли досягнуто ліміту

        ri = rj = None
        for key, r in list(routes.items()):
            if r[1] == i and r[-2] != j:
                ri = key
            if r[1] == j and r[-2] != i:
                rj = key
            if ri is not None and rj is not None:
                break

        if ri is not None and rj is not None and ri != rj:
            load_i = loads.get(ri, 0)
            load_j = loads.get(rj, 0)
            if load_i + load_j <= cap:
                new_route = routes[ri][:-1] + routes[rj][1:]
                new_key = min(ri, rj)
                del routes[ri], routes[rj], loads[ri], loads[rj]
                routes[new_key] = new_route
                loads[new_key] = load_i + load_j

    return list(routes.values())

    
def dist(i, j, coords):
    return math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
    
def greedy_initial(cap, coords, demands, depot=1):
    unvisited = set(coords.keys())
    unvisited.remove(depot)
    routes = []

    while unvisited:
        route = [depot]
        load = 0
        current = depot

        while True:
            candidates = [v for v in unvisited if load + demands[v] <= cap]
            if not candidates:
                break

            next_customer = min(candidates, key=lambda v: dist(current, v, coords))
            route.append(next_customer)
            load += demands[next_customer]
            unvisited.remove(next_customer)
            current = next_customer

        route.append(depot)
        routes.append(route)
    return routes
   
    
def two_opt(route, coords, first_improve=False, max_iter=10):
    """Повертає покращений (або той самий) маршрут після 2-opt."""
    best = route
    best_len = route_length(route, coords)
    n = len(route) - 1          # остання точка = depot (повторюється)

    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:   # ребра перетинаються — пропускаємо
                    continue
                new = route[:i] + route[i:j][::-1] + route[j:]
                newlen = best_len \
                       - eucl(coords[route[i-1]], coords[route[i]]) \
                       - eucl(coords[route[j-1]], coords[route[j]]) \
                       + eucl(coords[route[i-1]], coords[route[j-1]]) \
                       + eucl(coords[route[i]],   coords[route[j]])
                if newlen < best_len - 1e-6:   # знайшли краще
                    best, best_len = new, newlen
                    improved = True
                    if first_improve:
                        break
            if improved and first_improve:
                break
    return best
    
def inter_route_swap(routes, coords, cap, demands, depot=1):
    """
    Міняє клієнтів між різними маршрутами, якщо це зменшує довжину
    """
    for i in range(len(routes)):
        for j in range(i+1, len(routes)):
            r1, r2 = routes[i], routes[j]
            for a in range(1, len(r1) - 1):
                for b in range(1, len(r2) - 1):
                    n1, n2 = r1[a], r2[b]
                    d1 = sum(demands[n] for n in r1[1:-1]) - demands[n1] + demands[n2]
                    d2 = sum(demands[n] for n in r2[1:-1]) - demands[n2] + demands[n1]
                    if d1 > cap or d2 > cap:
                        continue
                    new_r1 = r1[:]
                    new_r2 = r2[:]
                    new_r1[a], new_r2[b] = n2, n1
                    old_len = route_length(r1, coords) + route_length(r2, coords)
                    new_len = route_length(new_r1, coords) + route_length(new_r2, coords)
                    if new_len < old_len:
                        routes[i], routes[j] = new_r1, new_r2
                        return True
    return False


def inter_route_relocate(routes, coords, cap, demands, depot=1):
    """
    Переміщує одного клієнта в інший маршрут, якщо це покращує довжину
    """
    for i in range(len(routes)):
        for j in range(len(routes)):
            if i == j:
                continue
            r1, r2 = routes[i], routes[j]
            for a in range(1, len(r1) - 1):
                node = r1[a]
                if sum(demands[n] for n in r2[1:-1]) + demands[node] > cap:
                    continue
                new_r1 = r1[:a] + r1[a+1:]
                for b in range(1, len(r2)):
                    new_r2 = r2[:b] + [node] + r2[b:]
                    old_len = route_length(r1, coords) + route_length(r2, coords)
                    new_len = route_length(new_r1, coords) + route_length(new_r2, coords)
                    if new_len < old_len:
                        routes[i], routes[j] = new_r1, new_r2
                        return True
    return False
