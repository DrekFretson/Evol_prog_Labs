"""ЛР №1, вариант 5: оптимизация функции Гриванка генетическим алгоритмом."""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

DIMENSION = 10
LOWER_BOUND = -600.0
UPPER_BOUND = 600.0


def griewank(x: list[float]) -> float:
    sum_part = sum(v * v for v in x) / 4000.0
    prod_part = 1.0
    for i, v in enumerate(x, start=1):
        prod_part *= math.cos(v / math.sqrt(i))
    return 1.0 + sum_part - prod_part


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    generations: int = 300
    crossover_probability: float = 0.90
    mutation_probability: float = 0.10
    mutation_sigma: float = 30.0
    tournament_size: int = 3
    elitism: int = 2

    @property
    def evaluation_budget(self) -> int:
        return self.population_size * (self.generations + 1)


def random_individual(rng: random.Random) -> list[float]:
    return [rng.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(DIMENSION)]


def tournament_selection(population: list[list[float]], fitnesses: list[float], rng: random.Random,
                         tournament_size: int) -> list[float]:
    indices = rng.sample(range(len(population)), tournament_size)
    best = min(indices, key=lambda idx: fitnesses[idx])
    return population[best][:]


def arithmetic_crossover(parent1: list[float], parent2: list[float], rng: random.Random) -> tuple[list[float], list[float]]:
    alpha = rng.random()
    child1 = [alpha * a + (1.0 - alpha) * b for a, b in zip(parent1, parent2)]
    child2 = [(1.0 - alpha) * a + alpha * b for a, b in zip(parent1, parent2)]
    return child1, child2


def gaussian_mutation(individual: list[float], rng: random.Random, probability: float, sigma: float) -> list[float]:
    child = individual[:]
    for i in range(DIMENSION):
        if rng.random() < probability:
            child[i] += rng.gauss(0.0, sigma)
            child[i] = max(LOWER_BOUND, min(UPPER_BOUND, child[i]))
    return child


def run_ga(config: GAConfig, seed: int) -> tuple[float, list[float], list[float]]:
    rng = random.Random(seed)
    population = [random_individual(rng) for _ in range(config.population_size)]
    fitnesses = [griewank(ind) for ind in population]
    history = [min(fitnesses)]

    for _ in range(config.generations):
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        new_population = [population[i][:] for i in ranked[:config.elitism]]

        while len(new_population) < config.population_size:
            p1 = tournament_selection(population, fitnesses, rng, config.tournament_size)
            p2 = tournament_selection(population, fitnesses, rng, config.tournament_size)

            if rng.random() < config.crossover_probability:
                c1, c2 = arithmetic_crossover(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]

            c1 = gaussian_mutation(c1, rng, config.mutation_probability, config.mutation_sigma)
            c2 = gaussian_mutation(c2, rng, config.mutation_probability, config.mutation_sigma)
            new_population.append(c1)
            if len(new_population) < config.population_size:
                new_population.append(c2)

        population = new_population
        fitnesses = [griewank(ind) for ind in population]
        history.append(min(fitnesses))

    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    return fitnesses[best_idx], population[best_idx][:], history


def random_search(evaluation_budget: int, seed: int) -> float:
    rng = random.Random(seed)
    best = math.inf
    for _ in range(evaluation_budget):
        best = min(best, griewank(random_individual(rng)))
    return best


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "best": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "worst": max(values),
    }


def save_runs_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_history_csv(path: Path, histories: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "min", "mean", "max"])
        for gen, vals in enumerate(zip(*histories)):
            writer.writerow([gen, min(vals), statistics.fmean(vals), max(vals)])


def plot_convergence(path: Path, histories: list[list[float]], title: str) -> None:
    mins, means, maxs = [], [], []
    for vals in zip(*histories):
        mins.append(min(vals)); means.append(statistics.fmean(vals)); maxs.append(max(vals))
    xs = range(len(means))
    plt.figure(figsize=(9, 5))
    plt.plot(xs, means, label="Среднее лучшее значение")
    plt.fill_between(xs, mins, maxs, alpha=0.2, label="min–max по запускам")
    plt.yscale("log")
    plt.xlabel("Поколение")
    plt.ylabel("Лучшее f(x), log scale")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_comparison(path: Path, groups: dict[str, list[float]]) -> None:
    plt.figure(figsize=(8, 5))
    plt.boxplot(list(groups.values()), tick_labels=list(groups.keys()))
    plt.yscale("log")
    plt.ylabel("Итоговое лучшее f(x), log scale")
    plt.title("Сравнение результатов независимых запусков")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def experiment(output_dir: Path, runs: int, base_seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Меняем только один фактор: вероятность мутации.
    config_a = GAConfig(mutation_probability=0.05)
    config_b = GAConfig(mutation_probability=0.20)
    configs = {"GA_mut_0.05": config_a, "GA_mut_0.20": config_b}

    rows: list[dict] = []
    all_results: dict[str, list[float]] = {}

    for name, config in configs.items():
        results, histories = [], []
        for run in range(runs):
            seed = base_seed + run
            best, vector, history = run_ga(config, seed)
            results.append(best); histories.append(history)
            rows.append({
                "method": name, "run": run + 1, "seed": seed, "best_fitness": best,
                "best_vector": " ".join(f"{v:.8f}" for v in vector),
                "evaluations": config.evaluation_budget,
            })
        all_results[name] = results
        save_history_csv(output_dir / f"convergence_{name}.csv", histories)
        plot_convergence(output_dir / f"convergence_{name}.png", histories, f"Сходимость: {name}")

    random_results = []
    for run in range(runs):
        seed = base_seed + run
        best = random_search(config_a.evaluation_budget, seed)
        random_results.append(best)
        rows.append({
            "method": "RandomSearch", "run": run + 1, "seed": seed, "best_fitness": best,
            "best_vector": "", "evaluations": config_a.evaluation_budget,
        })
    all_results["RandomSearch"] = random_results

    save_runs_csv(output_dir / "runs.csv", rows)
    plot_comparison(output_dir / "comparison.png", all_results)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("ЛР №1, вариант 5 — функция Гриванка\n")
        f.write(f"Независимых запусков: {runs}\n\n")
        for name, values in all_results.items():
            s = summarize(values)
            f.write(f"{name}\n")
            for key, value in s.items():
                f.write(f"  {key}: {value:.12g}\n")
            f.write("\n")

    print(f"Готово. Результаты сохранены в: {output_dir.resolve()}")
    for name, values in all_results.items():
        print(name, summarize(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР1 V5: ГА для функции Гриванка")
    parser.add_argument("--runs", type=int, default=20, help="число независимых запусков (по заданию >= 20)")
    parser.add_argument("--seed", type=int, default=2026, help="базовый seed")
    parser.add_argument("--output", type=Path, default=Path("results"), help="каталог результатов")
    args = parser.parse_args()
    if args.runs < 20:
        raise SystemExit("По методичке требуется не менее 20 независимых запусков.")
    experiment(args.output, args.runs, args.seed)


if __name__ == "__main__":
    main()
