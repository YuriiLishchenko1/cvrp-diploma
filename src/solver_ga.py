import math
import random
import time
import os
import streamlit as st
from utils.cvrp_parser import read_vrp

def load_cvrp_instance(file_path):
    """
    Wrapper around `read_vrp` to produce a full CVRP instance dict with distance matrix.
    """
    data = read_vrp(file_path)
    cap     = data["capacity"]
    coords  = data["coords"]
    demands = data["demand"]
    depot   = data["depot_id"]

    max_index = max(coords.keys())
    size = max_index + 1
    dist_matrix = [[0.0] * size for _ in range(size)]
    for i, (x1, y1) in coords.items():
        for j, (x2, y2) in coords.items():
            dist_matrix[i][j] = math.hypot(x2 - x1, y2 - y1) if i != j else 0.0

    return {
        "dimension": size,
        "capacity": cap,
        "num_vehicles": None,  # Optional field, can be set later
        "coords": coords,
        "demand": demands,
        "depot_id": depot,
        "dist_matrix": dist_matrix
    }

def create_initial_population(pop_size, customers):
    """Create initial population as a list of random permutations of customers."""
    population = []
    for _ in range(pop_size):
        chrom = customers.copy()
        random.shuffle(chrom)
        population.append(chrom)
    return population

def calculate_cost(chromosome, data):
    """Calculate total distance (cost) of the CVRP solution represented by the chromosome."""
    capacity = data["capacity"]
    demands = data["demand"]
    dist = data["dist_matrix"]
    depot = data["depot_id"]
    max_vehicles = data.get("num_vehicles", None)
    total_cost = 0.0
    current_load = 0
    last_node = depot
    vehicles_used = 1
    for cust in chromosome:
        if current_load + demands[cust] > capacity:
            # end current route and start a new one
            total_cost += dist[last_node][depot]
            vehicles_used += 1
            current_load = 0
            last_node = depot
        total_cost += dist[last_node][cust]
        current_load += demands[cust]
        last_node = cust
    total_cost += dist[last_node][depot]
    if max_vehicles is not None and vehicles_used > max_vehicles:
        total_cost += (vehicles_used - max_vehicles) * 1000000.0  # large penalty for extra vehicle
    return total_cost

def tournament_selection(population, costs, k=3):
    """Select one individual using tournament selection."""
    best_idx = None
    best_cost = float('inf')
    size = len(population)
    for _ in range(k):
        idx = random.randrange(size)
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best_idx = idx
    # Return a copy of the selected chromosome
    return population[best_idx][:]

def crossover(parent1, parent2):
    """Ordered crossover (OX) for two parent chromosomes."""
    size = len(parent1)
    a = random.randrange(size)
    b = random.randrange(size)
    if a > b:
        a, b = b, a
    child1 = [-1] * size
    child2 = [-1] * size
    # Copy segment from parents to offspring
    for i in range(a, b + 1):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
    # Fill remaining genes for child1 from parent2
    used1 = set(child1[a:b+1])
    j = (b + 1) % size
    k = (b + 1) % size
    while -1 in child1:
        gene = parent2[k]
        if gene not in used1:
            child1[j] = gene
            used1.add(gene)
            j = (j + 1) % size
        k = (k + 1) % size
    # Fill remaining genes for child2 from parent1
    used2 = set(child2[a:b+1])
    j = (b + 1) % size
    k = (b + 1) % size
    while -1 in child2:
        gene = parent1[k]
        if gene not in used2:
            child2[j] = gene
            used2.add(gene)
            j = (j + 1) % size
        k = (k + 1) % size
    return child1, child2

def mutate(chromosome, mut_prob):
    """Mutate a chromosome by swapping two genes with probability mut_prob."""
    if random.random() < mut_prob:
        size = len(chromosome)
        if size > 1:
            i = random.randrange(size)
            j = random.randrange(size)
            while j == i:
                j = random.randrange(size)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

def decode_routes(chromosome, data):
    """Decode a chromosome into a list of routes (each route is a list of nodes, including depot at start and end)."""
    depot = data["depot_id"]
    capacity = data["capacity"]
    demands = data["demand"]
    routes = []
    current_load = 0
    current_route = [depot]
    for cust in chromosome:
        if current_load + demands[cust] > capacity:
            current_route.append(depot)
            routes.append(current_route)
            current_route = [depot, cust]
            current_load = demands[cust]
        else:
            current_route.append(cust)
            current_load += demands[cust]
    current_route.append(depot)
    routes.append(current_route)
    return routes

def solve(file_path, sec_limit, pop_size, mut_prob):
    """Solve the CVRP using a Genetic Algorithm. Returns a result dictionary."""
    data = load_cvrp_instance(file_path)
    depot = data["depot_id"]
    customers = [node for node in data["demand"] if node != depot]
    customers.sort()
    # Initialize population and evaluate
    population = create_initial_population(pop_size, customers)
    costs = [calculate_cost(chrom, data) for chrom in population]
    best_idx = min(range(pop_size), key=lambda i: costs[i])
    best_cost = costs[best_idx]
    best_chrom = population[best_idx][:]
    start_time = time.time()
    # Evolution loop
    while time.time() - start_time < sec_limit:
        if st.session_state.get("stop_flag"):  # ⛔ якщо користувач натиснув "Зупинити"
            break
        new_population = []
        # Generate new_population via selection, crossover, mutation
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, costs)
            parent2 = tournament_selection(population, costs)
            if random.random() < 0.8:  # crossover probability
                offspring1, offspring2 = crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1[:], parent2[:]
            mutate(offspring1, mut_prob)
            mutate(offspring2, mut_prob)
            new_population.append(offspring1)
            if len(new_population) < pop_size:
                new_population.append(offspring2)
        new_costs = [calculate_cost(chrom, data) for chrom in new_population]
        # Track best and worst in new generation
        gen_best_idx = min(range(pop_size), key=lambda i: new_costs[i])
        gen_worst_idx = max(range(pop_size), key=lambda i: new_costs[i])
        if new_costs[gen_best_idx] < best_cost:
            best_cost = new_costs[gen_best_idx]
            best_chrom = new_population[gen_best_idx][:]
        # Elitism: carry over the global best to new population
        if best_cost < new_costs[gen_best_idx]:
            if best_chrom not in new_population:
                new_population[gen_worst_idx] = best_chrom[:]
                new_costs[gen_worst_idx] = best_cost
        population = new_population
        costs = new_costs
    # Decode best solution and prepare output
    best_routes = decode_routes(best_chrom, data)
    return {
        "file": os.path.basename(file_path),
        "algo": "Genetic Algorithm",
        "vehicles": len(best_routes),
        "distance": int(round(best_cost)),
        "time_sec": round(time.time() - start_time, 2),
        "routes": best_routes
    }
