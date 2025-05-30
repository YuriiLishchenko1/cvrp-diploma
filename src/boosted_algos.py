# ───────────────────────────────────────────
#  src/boosted_algos.py
# ───────────────────────────────────────────
from __future__ import annotations
import random, math
import time, json, pathlib, sys

from utils.cvrp_parser import read_vrp
from utils.constructive_ls import route_length, greedy_initial
from utils.local_move   import relocate_star     
import streamlit as st
# ────────────────────────────────────────────────────────

def ruin(routes: list[list[int]], k: int) -> tuple[list[list[int]], list[int]]:
    flat = [v for r in routes for v in r[1:-1]]
    rm = set(random.sample(flat, k))
    newR = []
    for r in routes:
        remain = [v for v in r if v not in rm]
        if len(remain) > 2:
            newR.append(remain)
    return newR, list(rm)

def fallback_insert(routes: list[list[int]],
                    v:      int,
                    cap:    int,
                    demand: dict[int,int],
                    coords: dict[int,tuple[float,float]]):
    depot = routes[0][0]
    for r in routes:
        load = sum(demand[x] for x in r[1:-1])
        if load + demand[v] <= cap:
            r.append(v)
            return
    routes.append([depot, v, depot])

def regret_insert_with_fallback(routes: list[list[int]],
                                pool:   list[int],
                                cap:    int,
                                demand: dict[int,int],
                                coords: dict[int,tuple[float,float]]):
    depot = routes[0][0]
    while pool:
        best, second = math.inf, math.inf
        best_r = None

        for v in pool:
            for ri, r in enumerate(routes):
                load = sum(demand[x] for x in r[1:-1])
                if load + demand[v] > cap:
                    continue
                for pos in range(1, len(r)):
                    add = (dist(r[pos-1], v, coords)
                         + dist(v, r[pos], coords)
                         - dist(r[pos-1], r[pos], coords))
                    if add < best:
                        second, best = best, add
                        best_r = (ri, pos, v)
                    elif add < second:
                        second = add

        if best_r:
            ri, pos, v = best_r
            routes[ri].insert(pos, v)
            pool.remove(v)
        else:
            v = pool.pop(0)
            fallback_insert(routes, v, cap, demand, coords)

def dist(a: int, b: int, C: dict[int,tuple[float,float]]) -> float:
    return math.hypot(C[a][0]-C[b][0], C[a][1]-C[b][1])

def enforce_capacity(routes: list[list[int]],
                     cap:    int,
                     demand: dict[int,int]) -> None:
    """Якщо якийсь маршрут перевантажений — винести клієнтів у нові маршрути."""
    depot = routes[0][0]
    i = 0
    # проходимо по списку, додаємо нові маршрути на ходу
    while i < len(routes):
        r = routes[i]
        load = sum(demand[x] for x in r[1:-1])
        if load <= cap:
            i += 1
            continue
        # поки перевантажений, виносимо останнього клієнта в новий маршрут
        v = r.pop()  
        # не забуваємо, що маршрут закінчується депо
        if v == depot:
            # якщо випадково вирвали депо — відновлюємо
            continue
        routes.append([depot, v, depot])
        # залишаємо i на місці, щоб перевірити ще раз цей маршрут
    # всі клієнти покриті, всі маршрути в межах cap

def solve_lns_boost(file_path: str|pathlib.Path, sec_limit: int = 30):
    cap, coords, dem = read_vrp(str(file_path))
    best = greedy_initial(cap, coords, dem)
    best_len = sum(route_length(r,coords) for r in best)

    psi_choices = (0.10, 0.15, 0.20)
    t0 = time.time()
    iter_ = 0

    while time.time() - t0 < sec_limit:
        if st.session_state.get("stop_flag"):  # ⛔ якщо користувач натиснув "Зупинити"
            break
        iter_ += 1
        ψ = random.choice(psi_choices)
        k = max(1, int(ψ * (len(coords)-1)))

        cur, pool = ruin([r[:] for r in best], k)
        regret_insert_with_fallback(cur, pool, cap, dem, coords)

        # ось тут перевіряємо й виправляємо capacity
        enforce_capacity(cur, cap, dem)

        relocate_star(cur, coords, dem, cap)

        s = sum(route_length(r,coords) for r in cur)
        if s < best_len:
            best, best_len = cur, s

    return {
        "file"     : pathlib.Path(file_path).name,
        "algo"     : "Boost-LNS",
        "vehicles" : len(best),
        "distance" : int(best_len),
        "time_sec" : round(time.time()-t0,2),
        "routes"   : best,
        "iter"     : iter_
    }

if __name__ == "__main__":
    fp  = sys.argv[1] if len(sys.argv)>1 else "data/cvrplib/A-n32-k5.vrp"
    sec = int(sys.argv[2]) if len(sys.argv)>2 else 30
    print(json.dumps(solve_lns_boost(fp, sec), indent=2))



# helper
def _pack(fp,algo,best_len,best_routes,t):
    return dict(file=pathlib.Path(fp).name,algo=algo,
                distance=int(best_len),vehicles=len(best_routes),
                time_sec=round(t,2),routes=best_routes)


def tabu_plus(fp, sec_limit=30, tenure_ratio=0.5):
    cap,coords,dem=read_vrp(str(fp)); depot=1
    cur=savings_initial(cap,coords,dem,depot)
    cur_len=sum(route_length(r,coords) for r in cur)
    best,best_len=list(cur),cur_len

    tabu=set(); tenure=int(len(coords)*tenure_ratio)
    t0=time.time(); iter=0
    while time.time()-t0<sec_limit:
        iter+=1
        best_nb,b_nb=len(cur)+1,1e9
        move_best=None

        # ------ neighbourhood: relocate + swap(2-2) ----------
        for ridx,r in enumerate(cur):
            for i in range(1,len(r)-1):
                for sidx,s in enumerate(cur):
                    for j in range(1,len(s)-1):
                        if ridx==sidx and abs(i-j)<=1: continue
                        # 2-2 swap
                        if i+1<len(r)-1 and j+1<len(s)-1:
                            mv=((r[i],r[i+1]),(s[j],s[j+1]))
                            if mv in tabu: continue
                            if _load_ok(r,s,i,j,dem,cap):
                                cand=_apply_22(cur,ridx,sidx,i,j)
                                clen=sum(route_length(x,coords) for x in cand)
                                if clen<b_nb:
                                    best_nb,b_nb=cand,clen; move_best=mv
                        # relocate
                        mv=(r[i],sidx)
                        if mv in tabu: continue
                        if dem[r[i]]+_load(s,dem,cap)<=cap:
                            cand=_apply_reloc(cur,ridx,sidx,i,j)
                            clen=sum(route_length(x,coords) for x in cand)
                            if clen<b_nb: best_nb,b_nb=cand,clen; move_best=mv

        cur,best_nb=list(best_nb),best_nb
        cur_len=b_nb
        tabu.add(move_best)
        if len(tabu)>tenure: tabu.pop()
        if cur_len<best_len: best,best_len=cur,cur_len

    return _pack(fp,"Tabu-plus",best_len,best,time.time()-t0)

# helper-utils relocate/swap omitted for brevity

def solve_ga_boost(fp, sec_limit=30, pop=60, mut=0.15):
    cap,coords,dem=read_vrp(str(fp)); depot=1
    nodes=[i for i in coords if i!=depot]
    L=len(nodes)

    # --- encode хромосому: проста permutaція вузлів -------------
    def split(chrom):
        routes=[]; load=0; r=[depot]
        for g in chrom:
            if load+dem[g]>cap:
                r.append(depot); routes.append(two_opt(r,coords))
                r=[depot]; load=0
            r.append(g); load+=dem[g]
        r.append(depot); routes.append(two_opt(r,coords))
        return routes

    def cost(ch): return sum(route_length(r,coords) for r in split(ch))

    # init pop: savings order + random perms
    base=[v for r in savings_initial(cap,coords,dem,depot) for v in r if v!=depot]
    P=[base[:]]+[random.sample(nodes,L) for _ in range(pop-1)]
    P.sort(key=cost)

    t0=time.time()
    while time.time()-t0<sec_limit:
        # --- selection (elit 20%) ---
        elite=P[:pop//5]
        # --- UXO crossover ---
        kids=[]
        while len(kids)<pop-len(elite):
            p1,p2=random.sample(P[:pop//2],2)
            cut1,cut2=sorted(random.sample(range(L),2))
            seg=p1[cut1:cut2]
            rest=[g for g in p2 if g not in seg]
            kid=rest[:cut1]+seg+rest[cut1:]
            # mutation (insert-swap)
            if random.random()<mut:
                a,b=random.sample(range(L),2); kid[a],kid[b]=kid[b],kid[a]
            # local LS boost
            kid=two_opt([depot]+kid+[depot],coords)[1:-1]
            kids.append(kid)
        P=sorted(elite+kids,key=cost)[:pop]

    best=P[0]; best_routes=split(best); best_len=cost(best)
    return _pack(fp,"GA-boost",best_len,best_routes,time.time()-t0)


def savings_tabu(fp, steps=5000):
    cap,coords,dem=read_vrp(str(fp)); depot=1
    cur=savings_initial(cap,coords,dem,depot)
    cur_len=sum(route_length(r,coords) for r in cur)
    best,best_len=list(cur),cur_len

    tabu=set(); tenure= len(coords)//2
    for _ in range(steps):
        # relocate move only (швидко)
        rids=[i for i,r in enumerate(cur) if len(r)>3]
        ridx=random.choice(rids); r=cur[ridx]
        pos=random.randint(1,len(r)-2)
        vid=r[pos]; cur[ridx].pop(pos)
        tidx=random.randint(0,len(cur)-1)
        t=cur[tidx]; ins_pos=random.randint(1,len(t)-1)
        cur[tidx].insert(ins_pos,vid)

        mv=(vid,tidx)
        if mv in tabu:
            # скасувати
            cur[tidx].pop(ins_pos); cur[ridx].insert(pos,vid)
            continue
        tabu.add(mv)
        if len(tabu)>tenure: tabu.pop()

        cur=[two_opt(r,coords) for r in cur]
        cur_len=sum(route_length(r,coords) for r in cur)
        if cur_len<best_len: best,best_len=list(cur),cur_len
    return _pack(fp,"Savings+Tabu",best_len,best,time.time())
