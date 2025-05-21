from __future__ import annotations
import math
from utils.constructive_ls import route_length

def dist(a: int, b: int, C):          # евклід. відстань
    return math.hypot(C[a][0] - C[b][0],
                      C[a][1] - C[b][1])

# ───────────────────────────────────────────────────────────────
def relocate_star(routes, coords, demand, cap):
    """
    Пробує перемістити один клієнт між (або усередині) маршрутів,
    приймаючи лише покращення. Повторює, доки є покращення.
    routes – list(list[int])   (дублюються вершини-депо на кінцях)
    """
    improved = True
    while improved:
        improved = False
        best_delta, best_move = 0, None

        for r1_idx, r1 in enumerate(routes):
            for pos1 in range(1, len(r1)-1):
                v = r1[pos1]
                load_r1 = sum(demand[x] for x in r1[1:-1])

                for r2_idx, r2 in enumerate(routes):
                    # залишок вантажу в r2
                    load_r2 = sum(demand[x] for x in r2[1:-1])
                    same_route = (r1_idx == r2_idx)

                    # куди вставити у r2
                    for pos2 in range(1, len(r2)):
                        if same_route and (pos2 == pos1 or pos2 == pos1+1):
                            continue   # немає сенсу

                        # перевірка місткості
                        if not same_route and load_r2 + demand[v] > cap:
                            continue

                        # різниця довжин після переміщення
                        d = 0
                        # 1. забираємо v з r1
                        d -= dist(r1[pos1-1], r1[pos1], coords)
                        d -= dist(r1[pos1],     r1[pos1+1], coords)
                        d += dist(r1[pos1-1], r1[pos1+1], coords)

                        # 2. вставляємо v у r2 між (pos2-1, pos2)
                        d -= dist(r2[pos2-1], r2[pos2], coords)
                        d += dist(r2[pos2-1], v, coords) + dist(v, r2[pos2], coords)

                        if d < best_delta:           # покращення!
                            best_delta = d
                            best_move  = (r1_idx, pos1, r2_idx, pos2, v)

        # реалізуємо найкращий рух
        if best_move:
            r1i, p1, r2i, p2, v = best_move
            routes[r1i].pop(p1)
            # якщо той самий маршрут і позиція після видалення зсунулася
            if r1i == r2i and p2 > p1:
                p2 -= 1
            routes[r2i].insert(p2, v)
            # якщо після видалення маршрут став 2-елементним (тільки депо) — видаляємо
            routes[:] = [r for r in routes if len(r) > 2]
            improved = True

    return routes
