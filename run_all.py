import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / "src"))

import csv, time, glob, pathlib, solver_baseline, solver_ga

files = glob.glob("data/cvrplib/*.vrp")[:10]   # перші 10
rows = []

for f in files:
    base = solver_baseline.solve(f, 10)  # 10 с
    ga   = solver_ga.solve_ga(f, sec_limit=120, pop_size=150)
    rows.append([pathlib.Path(f).name, base["distance"], ga["distance"]])

with open("outputs/results.csv", "w", newline="") as wf:
    csv.writer(wf).writerow(["file","ortools_dist","ga_dist"])
    csv.writer(wf).writerows(rows)
print("CSV saved → outputs/results.csv")
