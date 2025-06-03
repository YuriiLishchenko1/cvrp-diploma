import re

import math
import re

def read_vrp(path: str):
    with open(path) as f:
        lines = f.readlines()

    # Join all lines to search metadata
    full_text = "".join(lines)

    # Capacity
    capacity_match = re.search(r"CAPACITY\s*:\s*(\d+)", full_text)
    capacity = int(capacity_match.group(1)) if capacity_match else None

    vehicles_match = re.search(r"No of trucks\s*:\s*(\d+)", full_text, flags=re.IGNORECASE)
    num_vehicles = int(vehicles_match.group(1)) if vehicles_match else None


    # Dimension
    dimension_match = re.search(r"DIMENSION\s*:\s*(\d+)", full_text)
    dimension = int(dimension_match.group(1)) if dimension_match else None

    # Find sections
    coord_idx  = next(i for i, ln in enumerate(lines) if "NODE_COORD_SECTION" in ln)
    demand_idx = next(i for i, ln in enumerate(lines) if "DEMAND_SECTION" in ln)
    depot_idx  = next(i for i, ln in enumerate(lines) if "DEPOT_SECTION" in ln)

    # Parse coordinates
    coords = {}
    for ln in lines[coord_idx+1 : demand_idx]:
        if ln.strip().upper() in ("DEMAND_SECTION", "EOF"):
            break
        parts = ln.strip().split()
        if len(parts) >= 3:
            idx, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            coords[idx] = (x, y)

    # Parse demands
    demands = {}
    for ln in lines[demand_idx+1 : depot_idx]:
        if ln.strip().upper() in ("DEPOT_SECTION", "EOF"):
            break
        parts = ln.strip().split()
        if len(parts) >= 2:
            idx, dem = int(parts[0]), int(parts[1])
            demands[idx] = dem

    # Parse depot
    depot = None
    for ln in lines[depot_idx+1:]:
        ln = ln.strip()
        if ln == "-1" or ln.upper() == "EOF":
            break
        if ln.isdigit():
            depot = int(ln)
            break
    if depot is None:
        depot = 1  # fallback

    # Ensure depot has demand 0
    if depot in demands:
        demands[depot] = 0

    # Distance matrix
    max_id = max(coords.keys())
    dist_matrix = [[0.0 for _ in range(max_id + 1)] for _ in range(max_id + 1)]
    for i in coords:
        x1, y1 = coords[i]
        for j in coords:
            x2, y2 = coords[j]
            dist_matrix[i][j] = math.hypot(x1 - x2, y1 - y2) if i != j else 0.0

    return {
        "capacity": capacity,
        "coords": coords,
        "demand": demands,
        "depot_id": depot,
        "dist_matrix": dist_matrix,
        "num_vehicles": num_vehicles
    }

