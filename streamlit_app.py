# streamlit_app.py
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import streamlit as st
import matplotlib.pyplot as plt
import json
import pandas as pd
import io

from utils.io_runs import save_run, load_runs
from utils.cvrp_parser import read_vrp

from solver_baseline    import solve as solve_ort
from solver_ga          import solve as solve_ga
from solver_savings     import solve as solve_sv
from solver_tabu        import solve as solve_tb
from solver_sa          import solve as solve_sa
#from solver_fusion      import solve as solve_fusion
#from boosted_algos      import solve_lns_boost as solve_boost_lns
from solver_lns_sa      import solve as solve_lns_sa
from solver_lns         import solve as solve_lns
from solver_ga_boosted import solve as solve_ga_boosted
#from solver_tabu_boost import solve as solve_tabu_boost

DATA_DIR = ROOT / "data" / "cvrplib"

ALGO_FUNCS = {
    "OR-Tools": solve_ort,
    "Savings (Clarke–Wright)": solve_sv,
    "Genetic GA": solve_ga,
    "Simulated Annealing": solve_sa,
    "Tabu Search": solve_tb,
    "LNS":  solve_lns,
    "LNS-SA":  solve_lns_sa,
    #"Boost-LNS": solve_boost_lns,
    #"Fusion-R&R": solve_fusion,
    "Genetic GA Boosted": solve_ga_boosted,
    #"Tabu Boosted": solve_tabu_boost,
}

ALGO_TYPES = {
    "OR-Tools": "евристика",
    "Savings (Clarke–Wright)": "евристика",
    "Genetic GA": "метаевристика",
    "Simulated Annealing": "метаевристика",
    "Tabu Search": "метаевристика",
    "LNS": "метаевристика",
    "LNS-SA": "метаевристика",
    #"Boost-LNS": "гібрид",
    #"Fusion-R&R": "гібрид",
    "Genetic GA Boosted": "гібрид",
    #"Tabu Boosted": "гібрид",
}

st.sidebar.title("📦 CVRP Solver")

bench_files = sorted(f.name for f in DATA_DIR.glob("*.vrp"))
file_choice = st.sidebar.selectbox("Benchmark файл", bench_files)
sec_limit = st.sidebar.number_input("Час (сек)", min_value=5, max_value=600, value=30, step=1)

algo_name = st.sidebar.selectbox("Алгоритм (одноразово)", list(ALGO_FUNCS.keys()))
if algo_name in ["Genetic GA", "Genetic GA Boosted"]:
    pop_size = st.sidebar.number_input("Розмір популяції", 50, 500, 150, step=10)
    mut_prob = st.sidebar.slider("Ймовірність мутації", 0.0, 1.0, 0.2)
else:
    pop_size = mut_prob = None

if algo_name == "Simulated Annealing":
    init_temp = st.sidebar.slider("Початкова T", 1.0, 1000.0, 100.0)
    cool_rate = st.sidebar.slider("Коефіцієнт охолодження", 0.80, 0.99, 0.95)
else:
    init_temp = cool_rate = None

if algo_name == "OR-Tools":
    fs_strategy = st.sidebar.selectbox("FirstSolutionStrategy", ["SAVINGS", "PATH_CHEAPEST_ARC", "AUTOMATIC"])
else:
    fs_strategy = None


if st.sidebar.button("🚀 Розв'язати", disabled = False):
    st.session_state["stop_flag"] = False
    st.session_state["executing"] = True
    with st.status("Виконання алгоритму...", expanded=True) as status:
        st.button("🛑 Зупинити виконання", key="stop_button_single", on_click=lambda: st.session_state.update({"stop_flag": True}))
        fp = DATA_DIR / file_choice
        if algo_name == "OR-Tools":
            result = solve_ort(fp, sec_limit, fs_strategy)
        elif algo_name == "Genetic GA":
            result = solve_ga(fp, sec_limit, pop_size, mut_prob)
        elif algo_name == "Genetic GA Boosted":
            result = solve_ga_boosted(fp, sec_limit, pop_size, mut_prob)
        elif algo_name == "Savings (Clarke–Wright)":
            result = solve_sv(fp)
        elif algo_name == "Tabu Search":
            result = solve_tb(fp, sec_limit, tabu_tenure=10)
        elif algo_name == "Simulated Annealing":
            result = solve_sa(fp, sec_limit, init_temp, cool_rate)
        # elif algo_name == "Fusion-R&R":
            # result = solve_fusion(fp, sec_limit)
        # elif algo_name == "Boost-LNS":
            # result = solve_boost_lns(fp, sec_limit)
        elif algo_name == "LNS-SA":
            result = solve_lns_sa(fp, sec_limit)
        # elif algo_name == "Tabu Boosted":
            # result = solve_tabu_boost(fp, sec_limit)
        elif algo_name == "LNS":
            result = solve_lns(fp, sec_limit)
        else:
            st.error("Невідомий алгоритм")
            st.stop()
        status.update(label="✅ Завершено", state="complete")
        st.session_state["executing"] = False
        st.session_state["result"] = result

if st.sidebar.checkbox("Показати додаткові налаштування"):
    if st.sidebar.button("🏁 Запуск усіх алгоритмів"):
        st.session_state["stop_flag"] = False
        st.session_state["executing"] = True
        with st.status("Виконується порівняння алгоритмів...", expanded=True) as status:
            st.button("🛑 Зупинити виконання", key="stop_button_benchmark", on_click=lambda: st.session_state.update({"stop_flag": True}))
            fp = DATA_DIR / file_choice
            runs = []
            progress_bar = st.progress(0)
            total = len(ALGO_FUNCS)

            for idx, (name, func) in enumerate(ALGO_FUNCS.items(), 1):
                if st.session_state.get("stop_flag"):
                    st.warning("⛔ Виконання зупинено користувачем.")
                    break
                try:
                    status.update(label=f"🔄 Виконання: {name} ({idx}/{total})")
                    if name == "OR-Tools":
                        res = func(fp, sec_limit, None)
                    elif name in ["Genetic GA", "Genetic GA Boosted"]:
                        res = func(fp, sec_limit, 150, 0.2)
                    elif name == "Savings (Clarke–Wright)":
                        res = func(fp)
                    elif name == "Tabu Search":
                        res = func(fp, sec_limit, tabu_tenure=10)
                    elif name == "Simulated Annealing":
                        res = func(fp, sec_limit, 100.0, 0.95)
                    else:
                        res = func(fp, sec_limit)
                    res["algo"] = name
                    res["type"] = ALGO_TYPES.get(name, "?")
                    res["gap_%"] = round(((res["distance"] - res["optimal"]) / res["optimal"]) * 100, 2) if res.get("optimal") else None
                    res["distance"] = round(res["distance"], 2)
                    runs.append(res)
                except Exception as e:
                    st.warning(f"Алгоритм {name} згенерував помилку: {e}")
                progress_bar.progress(idx / total)

            st.session_state["benchmark"] = runs
            st.session_state["executing"] = False
            status.update(label="✅ Порівняння завершено", state="complete")


result = st.session_state.get("result")
result = st.session_state.get("result")
if result:
    st.subheader(f"📈 {result['file']} | 🚛 {result['vehicles']} | 📏 {result['distance']}")
    st.caption(f"⏱️ {result['time_sec']} с | Алгоритм: {algo_name}")

    show_result = st.checkbox("🔽 Показати деталі результату", value=True)
    if show_result:
        cap, coords, demands = read_vrp(DATA_DIR / file_choice)
        result["demand"] = demands
        result["coords"] = coords
        fig, ax = plt.subplots(figsize=(6, 6))
        xs, ys = zip(*coords.values())
        ax.scatter(xs, ys, c='black', s=10)

        for i, (x, y) in coords.items():
            ax.text(x, y, str(i), color=('red' if i == 1 else 'black'), fontsize=8)

        for ridx, route in enumerate(result["routes"]):
            rx, ry = zip(*(coords[i] for i in route))
            route_demand = sum(demands[i] for i in route if i != 1)
            ax.plot(rx, ry, linewidth=1.5, label=f"Route {ridx+1} [{route_demand}/{cap}]")

        ax.legend(fontsize=6)
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("📄 Результат JSON")
        st.json(result)
        json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8')
        st.download_button(
            "⬇️ Завантажити JSON",
            json_bytes,
            file_name=f"{result['file']}-{algo_name}-{result['distance']}km-{result['time_sec']}s.json",
            mime="application/json"
        )


bench = st.session_state.get("benchmark")
if bench:
    st.subheader("🏊 Результат порівняння")
    df = pd.DataFrame(bench)
    st.dataframe(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot.bar(x='algo', y='distance', ax=ax, legend=False)
    ax.set_ylabel("Distance")
    ax.set_xlabel("Algorithm")
    ax.set_title("Benchmark: Distances")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("⬇️ Завантажити графік PNG", buf.getvalue(), file_name="benchmark_chart.png", mime="image/png")

    latex = df.to_latex(index=False, float_format="%.2f")
    st.download_button("⬇️ Download Benchmark LaTeX", latex, file_name="benchmark.tex", mime="text/x-tex")

uploaded = st.sidebar.file_uploader("📂 Завантажте дані .JSON ", accept_multiple_files=True, type="json")
if uploaded:
    df_runs = load_runs(uploaded)
    df_runs["type"] = df_runs["algo"].map(ALGO_TYPES)
    st.subheader("📈 Порівняння")
    st.dataframe(df_runs.drop(columns=["iterations", "iter", "generations"], errors="ignore"))

    # for i, row in df_runs.iterrows():
        # with st.expander(f"🔍 {row['algo']} | 📏 {row['distance']} | ⏱️ {row['time_sec']} с"):
            # result = row.to_dict()
            # coords = result.get("coords")
            # demands = result.get("demand")
            # cap = result.get("capacity", 999)

            # if coords and demands:
                # coords = {int(k): tuple(v) for k, v in coords.items()}  # ключі з рядків у int
                # fig3, ax3 = plt.subplots(figsize=(6, 6))
                # xs, ys = zip(*coords.values())
                # ax3.scatter(xs, ys, c='black', s=10)

                # for idx, (x, y) in coords.items():
                    # ax3.text(x, y, str(idx), fontsize=8)

                # for ridx, route in enumerate(result.get("routes", [])):
                    # if all(i in coords for i in route):
                        # rx, ry = zip(*(coords[i] for i in route))
                        # route_demand = sum(demands.get(str(i), demands.get(i, 0)) for i in route if i != 1)
                        # ax3.plot(rx, ry, linewidth=1.5, label=f"Route {ridx+1} [{route_demand}/{cap}]")

                # ax3.legend(fontsize=6)
                # ax3.grid(True)
                # st.pyplot(fig3)
            # else:
                # st.warning("Неможливо відтворити маршрут: відсутні координати або попити")


    fig2, ax2 = plt.subplots()
    df_runs.plot.bar(x='algo', y='distance', ax=ax2, legend=False)
    ax2.set_ylabel("Distance")
    ax2.set_title("Comparison")
    st.pyplot(fig2)

    latex2 = df_runs.to_latex(index=False, float_format="%.2f")
    st.download_button("⬇️ Download Comparison LaTeX", latex2, file_name="compare.tex", mime="text/x-tex")
