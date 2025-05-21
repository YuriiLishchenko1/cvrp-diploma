import csv, glob, pathlib, pandas as pd
import matplotlib.pyplot as plt
from src import solver_baseline as sb, solver_ga as ga
from src import solver_savings as sv, solver_sweep as sw, solver_sa as sa

algos = {
  "ortools": lambda fp: sb.solve(fp)["distance"],
  "ga":      lambda fp: ga.solve_ga(fp,60,150)["distance"],
  "savings": lambda fp: sv.solve(fp)["distance"],
  "sweep":   lambda fp: sw.solve(fp)["distance"],
  "sa":      lambda fp: sa.solve(fp)["distance"],
}

def run_all(pattern="data/cvrplib/*.vrp"):
    rows=[]
    for fp in glob.glob(pattern):
        name=pathlib.Path(fp).name
        out={"file":name}
        for k,fn in algos.items():
            out[f"{k}_dist"] = fn(fp)
        print(out)
        rows.append(out)
    pathlib.Path("outputs").mkdir(exist_ok=True)
    with open("outputs/results.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    df=pd.DataFrame(rows)
    pathlib.Path("tables").mkdir(exist_ok=True)
    df.to_latex("tables/results_table.tex",index=False,caption="Сравнение",label="tab:res")
    import matplotlib.pyplot as plt
    pathlib.Path("figures").mkdir(exist_ok=True)
    df.set_index("file").plot.bar()
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("figures/results_distance.pdf")

if __name__=="__main__":
    run_all()
