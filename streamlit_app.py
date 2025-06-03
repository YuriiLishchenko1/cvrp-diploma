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
from solver_greedy      import solve as solve_gd
from solver_tabu        import solve as solve_tb
from solver_tabu_basic import solve as solve_tabu_basic
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
    "Greedy":solve_gd,
    "Genetic GA": solve_ga,
    "Genetic GA Boosted": solve_ga_boosted,
    "Simulated Annealing": solve_sa,
    "Tabu Search": solve_tb,
    "Tabu Search (Basic)": solve_tabu_basic,
    "LNS":  solve_lns,
    "LNS-SA":  solve_lns_sa,
    #"Boost-LNS": solve_boost_lns,
    #"Fusion-R&R": solve_fusion,
    #"Tabu Boosted": solve_tabu_boost,
}

ALGO_TYPES = {
    "OR-Tools": "Евристика",
    "Savings (Clarke–Wright)": "Евристика",
    "Greedy":"Евристика",
    "Genetic GA": "Метаевристика",
    "Simulated Annealing": "Метаевристика",
    "Tabu Search": "Метаевристика",
    "LNS": "Метаевристика",
    "LNS-SA": "гібрид",
    #"Boost-LNS": "гібрид",
    #"Fusion-R&R": "гібрид",
    "Genetic GA Boosted": "гібрид",
    #"Tabu Boosted": "гібрид",
    "Tabu Search (Basic)": "Метаевристика",
    
}

st.sidebar.title("📦 CVRP Solver")

run_single = st.sidebar.checkbox("Запуск одного алгоритму")
run_benchmark = st.sidebar.checkbox("Запустити тест алгоритмів")

bench_files = sorted(f.name for f in DATA_DIR.glob("*.vrp"))
file_choice = st.selectbox("Benchmark файл", bench_files)
sec_limit = st.number_input("Час (сек)", min_value=5, max_value=600, value=30, step=1)

if run_single:
    st.session_state.pop("benchmark", None)
    st.session_state.pop("json_view_file", None)
    st.session_state.pop("uploaded", None)
    st.subheader("⚙️ Налаштування одиночного запуску")

    algo_name = st.selectbox("Алгоритм", list(ALGO_FUNCS.keys()))

    pop_size = mut_prob = init_temp = cool_rate = fs_strategy = destroy_frac = None

    if algo_name in ["Genetic GA", "Genetic GA Boosted"]:
        pop_size = st.number_input("Розмір популяції", 50, 500, 150, step=10)
        mut_prob = st.slider("Ймовірність мутації", 0.0, 1.0, 0.2)

    if algo_name == "Simulated Annealing":
        init_temp = st.slider("Початкова T", 1.0, 1000.0, 100.0)
        cool_rate = st.slider("Коефіцієнт охолодження", 0.800, 0.999, 0.95)

    if algo_name == "OR-Tools":
        fs_strategy = st.selectbox("FirstSolutionStrategy", ["SAVINGS", "PATH_CHEAPEST_ARC", "AUTOMATIC"])

    if algo_name == "LNS-SA":
        if st.checkbox("Показати додаткові налаштування LNS-SA"):
            destroy_frac = st.slider("Частка руйнації (%)", 5, 50, 20) / 100
            init_temp = st.slider("Початкова температура (T0)", 10.0, 3000.0, 100.0)
            cool_rate = st.slider("Коефіцієнт охолодження", 0.90, 0.999, 0.995)
        else:
            destroy_frac = 0.2
            init_temp = 100.0
            cool_rate = 0.995

    if st.button("Розв'язати", disabled=False):
        st.session_state.pop("benchmark", None)
        st.session_state["stop_flag"] = False
        st.session_state["executing"] = True
        with st.status("Виконується алгоритм...", expanded=True) as status:
            st.button("Зупинити", key="stop_button_single", on_click=lambda: st.session_state.update({"stop_flag": True}))
            fp = DATA_DIR / file_choice
            func = ALGO_FUNCS[algo_name]
            try:
                if algo_name == "OR-Tools":
                    result = func(fp, sec_limit, fs_strategy)
                elif algo_name in ["Genetic GA", "Genetic GA Boosted"]:
                    result = func(fp, sec_limit, pop_size, mut_prob)
                elif algo_name == "Savings (Clarke–Wright)":
                    result = func(fp)
                elif algo_name == "Tabu Search":
                    result = func(fp, sec_limit, tabu_tenure=10)
                elif algo_name == "Simulated Annealing":
                    result = func(fp, sec_limit, init_temp, cool_rate)
                elif algo_name == "LNS-SA":
                    result = func(fp, sec_limit)
                else:
                    result = func(fp, sec_limit)
                result["algo"] = algo_name
                result["type"] = ALGO_TYPES.get(algo_name, "?")
                result["gap_%"] = round(((result["distance"] - result["optimal"]) / result["optimal"]) * 100, 2) if result.get("optimal") else None
                result["distance"] = round(result["distance"], 2)
                st.session_state["result"] = result
                status.update(label="Завершено", state="complete")
            except Exception as e:
                st.error(f"Помилка під час виконання алгоритму: {e}")
            st.session_state["executing"] = False

if run_benchmark:
    st.session_state.pop("result", None)
    st.session_state.pop("json_view_file", None)
    st.session_state.pop("uploaded", None)

    st.subheader("⚙️ Налаштування тесту алгоритмів")

    algo_types = ["Евристика", "Метаевристика", "гібрид"]
    selected_types = st.multiselect("Фільтр за типом:", algo_types, default=algo_types)

    filtered_algos = [name for name, t in ALGO_TYPES.items() if t in selected_types]

    if 'selected_algos' not in st.session_state:
        st.session_state.selected_algos = filtered_algos

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Обрати всі"):
            st.session_state.selected_algos = filtered_algos
    with col2:
        if st.button("Очистити вибір"):
            st.session_state.selected_algos = []

    selected_algos = st.multiselect(
        "Оберіть алгоритми для порівняння:",
        options=filtered_algos,
        default=st.session_state.selected_algos,
        key="selected_algos"
    )

    if st.button("Запуск обраних алгоритмів"):
        if len(selected_algos) < 2:
            st.warning("Потрібно обрати принаймні два алгоритми для порівняння.")
        else:
            st.session_state["stop_flag"] = False
            st.session_state.pop("result", None)
            st.session_state["executing"] = True

            with st.status("Виконується порівняння алгоритмів...", expanded=True) as status:
                st.button("Зупинити виконання", key="stop_button_benchmark", on_click=lambda: st.session_state.update({"stop_flag": True}))
                fp = DATA_DIR / file_choice
                runs = []
                progress_bar = st.progress(0)
                total = len(selected_algos)

                for idx, name in enumerate(selected_algos, 1):
                    if st.session_state.get("stop_flag"):
                        st.warning("Виконання зупинено користувачем.")
                        break
                    func = ALGO_FUNCS[name]
                    try:
                        status.update(label=f"Виконання: {name} ({idx}/{total})")
                        if name == "OR-Tools":
                            res = func(fp, sec_limit, "AUTOMATIC")
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
                status.update(label="Порівняння завершено", state="complete")


result = st.session_state.get("result")
if result:
    st.subheader(f"{result['file']} | {result['vehicles']} | {result['distance']}")
    st.caption(f"{result['time_sec']} с | Алгоритм: {algo_name}")

    show_result = st.checkbox("Показати деталі результату", value=True)
    if show_result:
        #cap, coords, demands = read_vrp(DATA_DIR / file_choice)
        data = read_vrp(DATA_DIR / file_choice)
        cap     = data["capacity"]
        coords  = data["coords"]
        demands = data["demand"]
        depot   = data["depot_id"]

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

        st.subheader("Результат JSON")
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
    st.subheader("Результат порівняння")
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

    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False)

    latex = df.to_latex(index=False, float_format="%.2f")

    json_buf = io.BytesIO()
    json_bytes = json.dumps(bench, ensure_ascii=False, indent=2).encode("utf-8")
    json_buf.write(json_bytes)
    json_buf.seek(0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button("завантажити PNG", buf.getvalue(), file_name="benchmark_chart.png", mime="image/png")
    with col2:
        st.download_button("завантажити Excel", excel_buf.getvalue(), file_name="benchmark.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col3:
        st.download_button("завантажити LaTeX", latex, file_name="benchmark.tex", mime="text/x-tex")
    with col4:
        st.download_button("завантажити JSON", json_buf.getvalue(), file_name="benchmark.json", mime="application/json")


if st.sidebar.checkbox(" Увімкнути завантаження JSON"):
    st.subheader("Завантаження та аналіз JSON результатів")

    tab1, tab2 = st.tabs(["Перегляд одного маршруту", "Порівняння кількох результатів"])

    with tab1:
        json_view_file = st.file_uploader("Оберіть JSON-файл для візуалізації маршруту", type="json", key="viewer")
        if json_view_file is not None:
            try:
                result = json.load(json_view_file)
                coords = result.get("coords")
                routes = result.get("routes")
                if not coords or not routes:
                    st.warning("У JSON відсутні ключі 'coords' або 'routes'.")
                else:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    xs, ys = zip(*coords.values())
                    ax.scatter(xs, ys, c='black', s=10)
                    for i, (x, y) in coords.items():
                        ax.text(x, y, str(i), color='red' if str(i) == "1" else 'black', fontsize=8)
                    for ridx, route in enumerate(routes):
                        rx, ry = zip(*(coords[str(i)] for i in route))
                        ax.plot(rx, ry, linewidth=1.5, label=f"Route {ridx+1}")
                    ax.legend(fontsize=6)
                    ax.grid(True)
                    ax.set_title(f'{result.get("file", "")} — {result.get("algo", "")}')
                    st.pyplot(fig)
                    st.success("Маршрут успішно візуалізовано.")
            except Exception as e:
                st.error(f"Помилка при обробці JSON: {e}")

    with tab2:
        uploaded = st.file_uploader("Завантажте кілька .JSON результатів для порівняння", accept_multiple_files=True, type="json", key="compare")
        if uploaded:
            df_runs = load_runs(uploaded)
            df_runs["type"] = df_runs["algo"].map(ALGO_TYPES)
            st.dataframe(df_runs.drop(columns=["iterations", "iter", "generations"], errors="ignore"))
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            df_runs.plot.bar(x='algo', y='distance', ax=ax2, legend=False)
            ax2.set_ylabel("Distance")
            ax2.set_title("Порівняння результатів")
            ax2.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig2)

            # Експорти
            col1, col2 = st.columns(2)

            with col1:
                latex2 = df_runs.to_latex(index=False, float_format="%.2f")
                st.download_button("⬇️ LaTeX таблиця", latex2, file_name="compare.tex", mime="text/x-tex")

            with col2:
                json_buf = io.BytesIO()
                json_bytes = json.dumps(df_runs.to_dict(orient="records"), ensure_ascii=False, indent=2).encode("utf-8")
                json_buf.write(json_bytes)
                json_buf.seek(0)
                st.download_button("⬇️ JSON", json_buf, file_name="compare.json", mime="application/json")
