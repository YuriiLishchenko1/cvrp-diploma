import json, time, csv, pathlib, itertools
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.utils.io_runs import save_run
from src.solver_ga     import solve as ga
from src.solver_sa     import solve as sa
from src.solver_tabu   import solve as tabu
from src.solver_savings import solve as sav
from src.solver_baseline import solve as ort
from src.solver_fusion import solve as solve_fusion
from src.solver_lns import solve as lns_sa
from src.solver_lns_sa import solve as lns
from src.boosted_algos import (
    solve_lns_boost, tabu_plus, solve_ga_boost, savings_tabu)


DATA = pathlib.Path("data/cvrplib")
ALGOS = {
    #"OR-Tools" : lambda fp: ort(fp, 60),
    #"Savings"  : lambda fp: sav(fp),
    #"Tabu"     : lambda fp: tabu(fp, 60),
    #"GA"       : lambda fp: ga(fp, 60, 150, 0.2),
    #"SA"       : lambda fp: sa(fp, 60),
    #"Fusion-R&R": lambda fp: solve_fusion(fp, 60),
    "LNS-SA" : lambda fp: lns_sa(fp, 60),   
    #"LNS"      : lambda fp: lns(fp, 60),
    #"Boost-LNS": lambda fp: solve_lns_boost(fp, 60),
    #"GA+": solve_ga_boost,
    #"Boost-LNS": solve_lns_boost,
    #"SaveTabu": savings_tabu,    
}

def run_all():
    out = pathlib.Path("runs")
    out.mkdir(exist_ok=True)
    for vrp in sorted(DATA.glob("P-*.vrp")):          # можна додати й B-, C-…
        for name, fn in ALGOS.items():
            t0 = time.time()
            res = fn(vrp)
            res["algo"] = name
            res["sec"]  = round(time.time()-t0,2)
            save_run(res)                        # → runs/{file}_{algo}.json
            print(vrp.name, name, res["distance"])

if __name__ == "__main__":
    run_all()
