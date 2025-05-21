from utils.constructive_ls import eucl, route_length

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