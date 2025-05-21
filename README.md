### Genetic Algorithm solver

```bash
python src/solver_ga.py data/cvrplib/A-n32-k5.vrp

streamlit run src/streamlit_app.py

python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt
python run_all.py          # запускає всі алгоритми, генерує таблицю/графік
streamlit run app.py       # UI
latexmk -pdf main.tex      # збірка диплома


"""
# CVRP‑Diploma

Дипломний проєкт — порівняння 6 класичних/метаевристичних алгоритмів CVRP.

## Запуск
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# batch‑benchmark
python run_all.py
# Streamlit інтерфейс
streamlit run streamlit_app.py
```
```

## Алгоритми
| ID | Метод | Модуль |
|----|-------|--------|
| ortools  | Google OR‑Tools GLS           | `solver_baseline.py` |
| savings  | Clarke‑Wright Savings         | `solver_savings.py`  |
| sweep    | Sweep + 2‑opt                | `solver_sweep.py`    |
| ls2opt   | 2‑opt / relocate (lib)       | `utils/constructive_ls.py` |
| sa       | Simulated Annealing          | `solver_sa.py`       |
| ga       | Genetic Algorithm            | `solver_ga.py`       |

Результати зберігаються у `outputs/results.csv`.
"""