"""ЛР №2, вариант 5.
Выбор комплекта оборудования для лаборатории при ограничениях бюджета,
мощности и совместимости. Ключевые части ГА реализованы без готовых GA-библиотек.
"""
from __future__ import annotations
import argparse, csv, random, statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

BUDGET = 2_000_000
POWER_LIMIT = 18_000
POPULATION_SIZE = 80
GENERATIONS = 250
RUNS = 20
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.03
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2
PENALTY_BUDGET = 0.00020
PENALTY_POWER = 0.020
PENALTY_CONFLICT = 45.0
BASE_SEED = 2505

@dataclass(frozen=True)
class Equipment:
    id: int
    name: str
    price: int
    power: int
    utility: int
    category: str

@dataclass
class Eval:
    utility: int
    price: int
    power: int
    conflicts: int
    feasible: bool
    fitness: float

def load_data():
    items=[]
    with open(DATA_DIR/"equipment.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            items.append(Equipment(int(r["id"]),r["name"],int(r["price_rub"]),int(r["power_w"]),
                                   int(r["utility"]),r["category"]))
    conflicts=[]
    with open(DATA_DIR/"conflicts.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            conflicts.append((int(r["item_a"])-1,int(r["item_b"])-1))
    return items, conflicts

def raw_metrics(chrom, items, conflicts):
    selected=[i for i,b in enumerate(chrom) if b]
    price=sum(items[i].price for i in selected)
    power=sum(items[i].power for i in selected)
    utility=sum(items[i].utility for i in selected)
    nconf=sum(1 for a,b in conflicts if chrom[a] and chrom[b])
    return utility,price,power,nconf

def evaluate(chrom, items, conflicts, method="repair"):
    utility,price,power,nconf=raw_metrics(chrom,items,conflicts)
    feasible=price<=BUDGET and power<=POWER_LIMIT and nconf==0
    if method=="penalty":
        penalty=max(0,price-BUDGET)*PENALTY_BUDGET + max(0,power-POWER_LIMIT)*PENALTY_POWER + nconf*PENALTY_CONFLICT
        fitness=utility-penalty
    else:
        fitness=float(utility) if feasible else -1e12
    return Eval(utility,price,power,nconf,feasible,fitness)

def repair(chrom, items, conflicts):
    """Детерминированно-случайный repair: сначала конфликты, затем ресурсы."""
    c=chrom[:]
    # Remove lower utility/resource-efficiency item from each active conflict.
    changed=True
    while changed:
        changed=False
        for a,b in conflicts:
            if c[a] and c[b]:
                sa=items[a].utility/(items[a].price/BUDGET + items[a].power/POWER_LIMIT + 1e-9)
                sb=items[b].utility/(items[b].price/BUDGET + items[b].power/POWER_LIMIT + 1e-9)
                c[a if sa < sb else b]=0
                changed=True
    while True:
        u,p,w,nc=raw_metrics(c,items,conflicts)
        if p<=BUDGET and w<=POWER_LIMIT and nc==0: break
        selected=[i for i,b in enumerate(c) if b]
        if not selected: break
        worst=min(selected,key=lambda i: items[i].utility/(items[i].price/BUDGET+items[i].power/POWER_LIMIT+1e-9))
        c[worst]=0
    return c

def random_chromosome(n, rng):
    return [1 if rng.random()<0.35 else 0 for _ in range(n)]

def tournament(pop, scores, rng):
    cand=rng.sample(range(len(pop)),TOURNAMENT_SIZE)
    return pop[max(cand,key=lambda i:scores[i])][:]

def one_point(a,b,rng):
    if len(a)<2: return a[:],b[:]
    p=rng.randrange(1,len(a))
    return a[:p]+b[p:], b[:p]+a[p:]

def uniform(a,b,rng):
    c1=[];c2=[]
    for x,y in zip(a,b):
        if rng.random()<0.5: c1.append(x);c2.append(y)
        else: c1.append(y);c2.append(x)
    return c1,c2

def mutate(c,rng):
    out=c[:]
    for i in range(len(out)):
        if rng.random()<MUTATION_RATE: out[i]=1-out[i]
    return out

def ga(items, conflicts, seed, constraint_method="repair", crossover_kind="uniform"):
    rng=random.Random(seed)
    pop=[random_chromosome(len(items),rng) for _ in range(POPULATION_SIZE)]
    if constraint_method=="repair": pop=[repair(c,items,conflicts) for c in pop]
    history=[]
    best_seen=None; best_eval=None
    for gen in range(GENERATIONS+1):
        ev=[evaluate(c,items,conflicts,constraint_method) for c in pop]
        scores=[x.fitness for x in ev]
        feasible=[(c,e) for c,e in zip(pop,ev) if e.feasible]
        if feasible:
            c,e=max(feasible,key=lambda ce:ce[1].utility)
            if best_eval is None or e.utility>best_eval.utility:
                best_seen=c[:];best_eval=e
        history.append(best_eval.utility if best_eval else 0)
        if gen==GENERATIONS: break
        elite_idx=sorted(range(len(pop)),key=lambda i:scores[i],reverse=True)[:ELITE_COUNT]
        new=[pop[i][:] for i in elite_idx]
        while len(new)<POPULATION_SIZE:
            p1=tournament(pop,scores,rng); p2=tournament(pop,scores,rng)
            if rng.random()<CROSSOVER_RATE:
                c1,c2=(uniform(p1,p2,rng) if crossover_kind=="uniform" else one_point(p1,p2,rng))
            else: c1,c2=p1[:],p2[:]
            for child in (mutate(c1,rng),mutate(c2,rng)):
                if constraint_method=="repair": child=repair(child,items,conflicts)
                new.append(child)
                if len(new)>=POPULATION_SIZE: break
        pop=new
    return best_seen,best_eval,history


def random_feasible_baseline(items, conflicts, seed, evaluation_budget):
    """Случайный допустимый поиск при том же числе оценок кандидатов, что и один запуск ГА."""
    rng = random.Random(seed)
    best_c = [0] * len(items)
    best_e = evaluate(best_c, items, conflicts, "repair")
    accepted = 0
    attempts = 0
    max_attempts = evaluation_budget * 30
    while accepted < evaluation_budget and attempts < max_attempts:
        attempts += 1
        c = [1 if rng.random() < 0.12 else 0 for _ in items]
        e = evaluate(c, items, conflicts, "repair")
        if not e.feasible:
            continue
        accepted += 1
        if e.utility > best_e.utility:
            best_c, best_e = c[:], e
    return best_c, best_e

def greedy_baseline(items, conflicts):
    order=sorted(range(len(items)),key=lambda i:items[i].utility/(items[i].price/BUDGET+items[i].power/POWER_LIMIT),reverse=True)
    c=[0]*len(items)
    for i in order:
        trial=c[:];trial[i]=1
        if evaluate(trial,items,conflicts,"repair").feasible: c=trial
    return c,evaluate(c,items,conflicts,"repair")

def save_best(name, chrom, ev, items):
    with open(RESULTS_DIR/f"best_{name}.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["id","name","price_rub","power_w","utility","category"])
        for i,b in enumerate(chrom or []):
            if b:
                x=items[i];w.writerow([x.id,x.name,x.price,x.power,x.utility,x.category])
        w.writerow([])
        w.writerow(["TOTAL","",ev.price if ev else 0,ev.power if ev else 0,ev.utility if ev else 0,""])

def run_experiments():
    RESULTS_DIR.mkdir(exist_ok=True)
    items,conflicts=load_data()
    configs=[
        ("GA_repair_uniform","repair","uniform"),
        ("GA_penalty_uniform","penalty","uniform"),
        ("GA_repair_onepoint","repair","onepoint"),
    ]
    all_rows=[]; histories={k:[] for k,_,_ in configs}; best_global={}
    for name,method,cross in configs:
        for r in range(RUNS):
            seed=BASE_SEED+r
            c,e,h=ga(items,conflicts,seed,method,cross)
            histories[name].append(h)
            if e:
                all_rows.append([name,r+1,seed,e.utility,e.price,e.power,e.conflicts,int(e.feasible)])
                if name not in best_global or e.utility>best_global[name][1].utility: best_global[name]=(c,e)
    gc,ge=greedy_baseline(items,conflicts)
    all_rows.append(["Greedy",1,BASE_SEED,ge.utility,ge.price,ge.power,ge.conflicts,1])

    # Fair stochastic baseline: same candidate-evaluation budget as GA:
    # initial population + one population per generation.
    evaluation_budget = POPULATION_SIZE * (GENERATIONS + 1)
    random_baseline_values = []
    best_random = None
    for r in range(RUNS):
        seed = BASE_SEED + r
        rc, re = random_feasible_baseline(items, conflicts, seed, evaluation_budget)
        random_baseline_values.append(re.utility)
        all_rows.append(["RandomFeasibleSearch", r+1, seed, re.utility, re.price, re.power, re.conflicts, 1])
        if best_random is None or re.utility > best_random[1].utility:
            best_random = (rc, re)

    with open(RESULTS_DIR/"runs.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["method","run","seed","utility","price_rub","power_w","conflicts","feasible"]);w.writerows(all_rows)

    lines=["ЛР №2, вариант 5 — выбор оборудования",f"Бюджет: {BUDGET} руб.",f"Лимит мощности: {POWER_LIMIT} Вт",f"Независимых запусков ГА: {RUNS}",""]
    for name,_,_ in configs:
        vals=[r[3] for r in all_rows if r[0]==name]
        lines += [name,f"  best: {max(vals):.6g}",f"  mean: {statistics.mean(vals):.6g}",f"  median: {statistics.median(vals):.6g}",
                  f"  std: {statistics.stdev(vals):.6g}",f"  worst: {min(vals):.6g}",""]
        save_best(name,*best_global[name],items)
    lines += ["Greedy baseline (дополнительный конструктивный ориентир)",f"  utility: {ge.utility}",f"  price: {ge.price}",f"  power: {ge.power}",""]
    lines += ["RandomFeasibleSearch (тот же бюджет оценок, что у ГА)",f"  evaluations per run: {evaluation_budget}",
              f"  best: {max(random_baseline_values):.6g}",f"  mean: {statistics.mean(random_baseline_values):.6g}",
              f"  median: {statistics.median(random_baseline_values):.6g}",
              f"  std: {statistics.stdev(random_baseline_values):.6g}",
              f"  worst: {min(random_baseline_values):.6g}"]
    save_best("RandomFeasibleSearch", *best_random, items)
    (RESULTS_DIR/"summary.txt").write_text("\n".join(lines),encoding="utf-8")

    # Convergence
    plt.figure(figsize=(10,6))
    for name,_,_ in configs:
        hs=histories[name]
        means=[statistics.mean(h[g] for h in hs) for g in range(GENERATIONS+1)]
        plt.plot(range(GENERATIONS+1),means,label=name)
    plt.xlabel("Поколение");plt.ylabel("Средняя лучшая полезность")
    plt.title("Сходимость ГА по 20 независимым запускам");plt.grid(alpha=.25);plt.legend();plt.tight_layout()
    plt.savefig(RESULTS_DIR/"convergence.png",dpi=160);plt.close()

    # Boxplot
    labels=[x[0] for x in configs] + ["RandomFeasibleSearch"]
    data=[[r[3] for r in all_rows if r[0]==name] for name in labels]
    plt.figure(figsize=(10,6));plt.boxplot(data,tick_labels=labels)
    plt.axhline(ge.utility,linestyle="--",label=f"Greedy = {ge.utility}")
    plt.ylabel("Полезность");plt.title("Сравнение алгоритмических вариантов");plt.xticks(rotation=12);plt.legend();plt.tight_layout()
    plt.savefig(RESULTS_DIR/"comparison.png",dpi=160);plt.close()

    # Required feasible/infeasible examples
    examples=[
        ("feasible_1",[1 if i in [0,1,2,3] else 0 for i in range(len(items))]),
        ("feasible_2",[1 if i in [4,7,8,14] else 0 for i in range(len(items))]),
        ("infeasible_budget",[1]*len(items)),
        ("infeasible_conflict",[1 if i in [5,26] else 0 for i in range(len(items))]),
    ]
    with open(RESULTS_DIR/"examples.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["example","utility","price_rub","power_w","conflicts","feasible","selected_ids"])
        for n,c in examples:
            e=evaluate(c,items,conflicts,"repair")
            w.writerow([n,e.utility,e.price,e.power,e.conflicts,int(e.feasible),";".join(str(i+1) for i,b in enumerate(c) if b)])
    print("\n".join(lines))
    print(f"\nРезультаты сохранены: {RESULTS_DIR}")

def main():
    p=argparse.ArgumentParser(description="ЛР2, вариант 5: дискретный ГА выбора оборудования")
    p.add_argument("--runs",action="store_true",help="выполнить полную серию экспериментов")
    args=p.parse_args()
    run_experiments()

if __name__=="__main__":
    main()
