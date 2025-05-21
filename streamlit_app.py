# streamlit_app.py (D:\cvrp-diploma\)
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import streamlit as st
import matplotlib.pyplot as plt
import json, io
from utils.io_runs import save_run, load_runs
from utils.cvrp_parser import read_vrp
from solver_baseline import solve as solve_ort
from solver_ga import solve as solve_ga
from solver_savings import solve as solve_sv
from solver_tabu import solve as solve_tb
from solver_sa import solve as solve_sa
from solver_fusion import solve as solve_fusion
from boosted_algos import solve_lns_boost as solve_boost_lns

DATA_DIR = ROOT / "data" / "cvrplib"

ALGO_FUNCS = {
    "OR-Tools": solve_ort,
    "Genetic GA": solve_ga,
    "Savings (Clarke–Wright)": solve_sv,
    "Tabu Search": solve_tb,
    "Simulated Annealing": solve_sa,
    "Fusion-R&R": solve_fusion,
    "Boost-LNS": solve_boost_lns,
}

st.sidebar.title("📦 CVRP Solver")
bench_files = sorted(f.name for f in DATA_DIR.glob("*.vrp"))

file_choice = st.sidebar.selectbox("Benchmark файл", bench_files)
algo_name = st.sidebar.selectbox("Алгоритм", list(ALGO_FUNCS.keys()))
sec_limit = st.sidebar.slider("Час (сек)", 5, 120, 30)

if algo_name == "Genetic GA":
    pop_size = st.sidebar.number_input("Розмір популяції", 50, 500, 150, step=10)
    mut_prob = st.sidebar.slider("Ймовірність мутації", 0.0, 1.0, 0.2)
else:
    pop_size = mut_prob = None

if algo_name == "Simulated Annealing":
    init_temp = st.sidebar.slider("Початкова T", 1.0, 1000.0, 100.0)
    cool_rate = st.sidebar.slider("Коефіцієнт охолодження", 0.80, 0.99, 0.95)

if algo_name == "OR-Tools":
    fs_strategy = st.sidebar.selectbox("FirstSolutionStrategy", ["SAVINGS", "PATH_CHEAPEST_ARC", "AUTOMATIC"])

if st.sidebar.button("🚀 Розв'язати"):
    fp = DATA_DIR / file_choice
    if algo_name == "OR-Tools":
        result = solve_ort(fp, sec_limit, fs_strategy)
    elif algo_name == "Genetic GA":
        result = solve_ga(fp, sec_limit, pop_size, mut_prob)
    elif algo_name == "Savings (Clarke–Wright)":
        result = solve_sv(fp)
    elif algo_name == "Tabu Search":
        result = solve_tb(fp, sec_limit, tabu_tenure=10)
    elif algo_name == "Simulated Annealing":
        result = solve_sa(fp, sec_limit, init_temp, cool_rate)
    elif algo_name == "Fusion-R&R":
        result = solve_fusion(fp, sec_limit)
    elif algo_name == "Boost-LNS":
        result = solve_boost_lns(fp, sec_limit)
    else:
        st.error("Невідомий алгоритм")
        st.stop()
    st.session_state["result"] = result

result = st.session_state.get("result")

if st.sidebar.button("💾 Зберегти результат"):
    run = result.copy()
    run["algo"] = algo_name
    save_run(run)
    st.success(f"Результат алгоритму «{algo_name}» збережено.")

st.title("🚚 CVRP Solver Playground")

if result:
    st.subheader(f"🗄️ {result['file']} | 🚛 {result['vehicles']} | 📏 {result['distance']} км")
    st.caption(f"⏱️ {result['time_sec']} с | Алгоритм: {algo_name}")

    cap, coords, _ = read_vrp(DATA_DIR / file_choice)
    fig, ax = plt.subplots(figsize=(6, 6))
    xs, ys = zip(*coords.values())
    ax.scatter(xs, ys, c='black', s=10)

    for i, (x, y) in coords.items():
        color = 'red' if i == 1 else 'black'
        ax.text(x, y, str(i), color=color, fontsize=8)

    for ridx, route in enumerate(result["routes"]):
        rx, ry = zip(*(coords[i] for i in route))
        ax.plot(rx, ry, linewidth=1.5, label=f"Route {ridx+1}")

    ax.legend(fontsize=6)
    ax.set_title("Routes on Cartesian plane")
    ax.grid(True)
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    st.download_button("⬇️ Download PNG", buf.getvalue(), file_name="routes.png", mime="image/png")

    st.subheader("📄 Result JSON")
    st.json(result)
    json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8')
    st.download_button(
        "⬇️ Download JSON", 
        json_bytes, 
        file_name=f"{result['file']}-{algo_name}-{result['distance']}km-{result['time_sec']}s.json",
        mime="application/json"
    )


uploaded = st.sidebar.file_uploader("📂 Load run JSON files", accept_multiple_files=True, type="json")
if uploaded:
    df_runs = load_runs(uploaded)
    st.subheader("🏁 Порівняння алгоритмів")
    st.dataframe(df_runs)

    fig, ax = plt.subplots()
    df_runs.plot.bar(x='algo', y='distance', ax=ax, legend=False)
    ax.set_ylabel("Distance")
    ax.set_title("Comparison of Algorithm Results")
    st.pyplot(fig)

    latex = df_runs.to_latex(index=False)
    st.download_button("⬇️ Download LaTeX", latex, file_name="compare.tex", mime="text/x-tex")
