# файл: src/utils/constructive_ls.py

from utils.cvrp_parser import read_vrp
from math import hypot
from collections import defaultdict
import math
# --- ваші вже наявні функції ---
def eucl(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])

def route_length(route, coords):
    dist = 0
    for i,j in zip(route[:-1], route[1:]):
        dist += eucl(coords[i], coords[j])
    return dist

# --- нова функція savings_initial ---
def savings_initial(cap, coords, demands, depot=1):
    """
    Алгоритм Clarke–Wright Savings із захистом від KeyError.
    Повертає список маршрутів [[depot, ..., depot], ...].
    """
    # побудова початкових маршрутів: для кожного клієнта окремий маршрут
    routes = {i: [depot, i, depot] for i in coords if i != depot}
    loads  = {i: demands[i] for i in coords if i != depot}
    # обчислити економію для кожної пари
    savings = []
    for i in coords:
        for j in coords:
            if i < j and i != depot and j != depot:
                d_ij = eucl(coords[i], coords[j])
                d_id = eucl(coords[i], coords[depot])
                d_jd = eucl(coords[j], coords[depot])
                savings.append((i, j, d_id + d_jd - d_ij))
    savings.sort(key=lambda x: x[2], reverse=True)

    for i, j, s in savings:
        # знайти маршрути, які містять i і j
        ri = rj = None
        for key, r in list(routes.items()):
            if r[1] == i and r[-2] != j:  # i на початку
                ri = key
            if r[1] == j and r[-2] != i:
                rj = key
            if ri is not None and rj is not None:
                break

        # якщо i і j в різних маршрутах і можна об’єднати за місткістю
        if ri is not None and rj is not None and ri != rj:
            load_i = loads.get(ri, 0)
            load_j = loads.get(rj, 0)
            if load_i + load_j <= cap:
                # об’єднати: drop кінцевий depot із ri і початковий із rj
                new_route = routes[ri][:-1] + routes[rj][1:]
                new_key = min(ri, rj)
                # очистити старі
                del routes[ri]
                del routes[rj]
                del loads[ri]
                del loads[rj]
                # додати новий
                routes[new_key] = new_route
                loads[new_key] = load_i + load_j

    return list(routes.values())
    
def dist(i, j, coords):
    return math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
    
def greedy_initial(cap, coords, demands, depot=1):
    """
    Створює початкові маршрути простим greedy методом:
    кожен наступний клієнт — найближчий, який поміщається по місткості.
    """
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

