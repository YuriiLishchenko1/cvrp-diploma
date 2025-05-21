import pandas as pd, pathlib, textwrap
from src.utils.io_runs import load_runs

df = load_runs(pathlib.Path("runs").glob("*.json"))
pivot = (df
         .groupby(["file","algo"],as_index=False)["distance"]
         .min()                       # брати найкращий run (якщо буде кілька)
         .pivot(index="file", columns="algo", values="distance")
         .sort_index())
latex = pivot.to_latex(float_format="%.0f")
(Path("tables")/ "results.tex").write_text(
    textwrap.dedent(r"""
    % --- auto-generated ---
    \begin{table}[H]\centering
    \caption{Результати CVRP на наборах A-n}
    """ + latex + r"\end{table}")
)
print("LaTeX table saved → tables/results.tex")
