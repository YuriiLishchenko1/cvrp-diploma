# src/utils/cvrp_parser.py
import re

def read_vrp(path: str):
    with open(path) as f:
        lines = f.readlines()

    capacity = int(re.search(r"CAPACITY\s*:\s*(\d+)", "".join(lines)).group(1))

    # знайти позиції секцій — шукаємо по startswith
    coord_idx = next(i for i,ln in enumerate(lines) if ln.startswith("NODE_COORD_SECTION"))
    demand_idx = next(i for i,ln in enumerate(lines) if ln.startswith("DEMAND_SECTION"))
    depot_idx  = next(i for i,ln in enumerate(lines) if ln.startswith("DEPOT_SECTION"))

    coords = {}
    for ln in lines[coord_idx+1 : demand_idx]:
        if ln.strip().upper() in ("DEMAND_SECTION", "EOF"):
            break
        idx, x, y = map(float, ln.split())
        coords[int(idx)] = (x, y)

    demands = {}
    for ln in lines[demand_idx+1 : depot_idx]:
        if ln.strip().upper() in ("DEPOT_SECTION", "EOF"):
            break
        idx, dem = map(int, ln.split())
        demands[idx] = dem

    return capacity, coords, demands
