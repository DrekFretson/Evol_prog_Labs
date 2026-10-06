import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Лабораторная работа №1
# Вариант 19
# Функция HappyCat
# ============================================================

VARIANT = 19
DIMENSION = 6

LOWER_BOUND = -20.0
UPPER_BOUND = 20.0

THEORETICAL_MINIMUM = 0.0
THEORETICAL_MINIMIZER = [-1.0] * DIMENSION


@dataclass
class GAConfig:
    population_size: int
    generations: int
    crossover_probability: float
    mutation_probability: float
    mutation_sigma: float
    tournament_size: int
    elitism: int

    @property
    def evaluation_budget(self) -> int:
        """
        Начальная популяция +
        population_size вычислений на каждое поколение.
        """
        return self.population_size * (self.generations + 1)


# ============================================================
# Целевая функция
# ============================================================

def happy_cat(x):
    """
    Функция HappyCat для варианта 19.

    f(x) =
        |sum(x_i^2) - d|^(1/4)
        +
        (0.5 * sum(x_i^2) + sum(x_i)) / d
        +
        0.5
    """

    sum_squares = sum(value ** 2 for value in x)
    sum_values = sum(x)

    return (
        abs(sum_squares - DIMENSION) ** 0.25
        + (0.5 * sum_squares + sum_values) / DIMENSION
        + 0.5
    )


# ============================================================
# Создание особей
# ============================================================

def create_individual(rng):
    """
    Создание случайной допустимой особи.
    """

    return [
        rng.uniform(LOWER_BOUND, UPPER_BOUND)
        for _ in range(DIMENSION)
    ]


def create_population(size, rng):
    return [
        create_individual(rng)
        for _ in range(size)
    ]


# ============================================================
# Селекция
# ============================================================

def tournament_selection(
    population,
    fitnesses,
    tournament_size,
    rng,
):
    """
    Турнирная селекция.

    Из случайно выбранных tournament_size особей
    выбирается особь с минимальным значением функции.
    """

    indices = rng.sample(
        range(len(population)),
        tournament_size,
    )

    best_index = min(
        indices,
        key=lambda index: fitnesses[index],
    )

    return population[best_index][:]


# ============================================================
# Кроссовер
# ============================================================

def arithmetic_crossover(parent1, parent2, rng):
    """
    Арифметический кроссовер.

    Потомки являются линейными комбинациями родителей.
    """

    alpha = rng.random()

    child1 = []
    child2 = []

    for value1, value2 in zip(parent1, parent2):
        child1.append(
            alpha * value1
            + (1.0 - alpha) * value2
        )

        child2.append(
            (1.0 - alpha) * value1
            + alpha * value2
        )

    return child1, child2


# ============================================================
# Мутация
# ============================================================

def gaussian_mutation(
    individual,
    mutation_probability,
    sigma,
    rng,
):
    """
    Гауссовская мутация.

    Каждая координата мутирует независимо.
    После мутации координата ограничивается диапазоном [-20, 20].
    """

    result = individual[:]

    for i in range(DIMENSION):

        if rng.random() < mutation_probability:

            result[i] += rng.gauss(
                0.0,
                sigma,
            )

            result[i] = max(
                LOWER_BOUND,
                min(
                    UPPER_BOUND,
                    result[i],
                ),
            )

    return result


# ============================================================
# Генетический алгоритм
# ============================================================

def run_ga(config, seed):
    """
    Один запуск генетического алгоритма.

    Возвращает:
    - лучшее значение функции;
    - лучший вектор;
    - историю лучшего значения по поколениям.
    """

    rng = random.Random(seed)

    population = create_population(
        config.population_size,
        rng,
    )

    fitnesses = [
        happy_cat(individual)
        for individual in population
    ]

    history = [
        min(fitnesses)
    ]

    for generation in range(config.generations):

        # ----------------------------------------------------
        # Элитизм
        # ----------------------------------------------------

        sorted_indices = sorted(
            range(len(population)),
            key=lambda index: fitnesses[index],
        )

        new_population = []

        for index in sorted_indices[:config.elitism]:
            new_population.append(
                population[index][:]
            )

        # ----------------------------------------------------
        # Создание остальных потомков
        # ----------------------------------------------------

        while len(new_population) < config.population_size:

            parent1 = tournament_selection(
                population,
                fitnesses,
                config.tournament_size,
                rng,
            )

            parent2 = tournament_selection(
                population,
                fitnesses,
                config.tournament_size,
                rng,
            )

            # Кроссовер
            if rng.random() < config.crossover_probability:

                child1, child2 = arithmetic_crossover(
                    parent1,
                    parent2,
                    rng,
                )

            else:

                child1 = parent1[:]
                child2 = parent2[:]

            # Мутация
            child1 = gaussian_mutation(
                child1,
                config.mutation_probability,
                config.mutation_sigma,
                rng,
            )

            child2 = gaussian_mutation(
                child2,
                config.mutation_probability,
                config.mutation_sigma,
                rng,
            )

            new_population.append(child1)

            if len(new_population) < config.population_size:
                new_population.append(child2)

        population = new_population

        fitnesses = [
            happy_cat(individual)
            for individual in population
        ]

        history.append(
            min(fitnesses)
        )

    best_index = min(
        range(len(population)),
        key=lambda index: fitnesses[index],
    )

    best_vector = population[best_index][:]
    best_fitness = fitnesses[best_index]

    return (
        best_fitness,
        best_vector,
        history,
    )


# ============================================================
# Случайный поиск
# ============================================================

def random_search(evaluation_budget, seed):
    """
    Random Search при том же количестве
    вычислений функции, что и ГА.
    """

    rng = random.Random(seed)

    best_value = math.inf
    best_vector = None

    for _ in range(evaluation_budget):

        individual = create_individual(rng)

        fitness = happy_cat(individual)

        if fitness < best_value:

            best_value = fitness
            best_vector = individual[:]

    return best_value, best_vector


# ============================================================
# Статистика
# ============================================================

def calculate_statistics(values):

    return {
        "best": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "worst": max(values),
    }


# ============================================================
# CSV
# ============================================================

def save_runs_csv(path, rows):

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "run",
                "seed",
                "best_fitness",
                "best_vector",
                "evaluations",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def save_convergence_csv(path, histories):

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "generation",
                "min",
                "mean",
                "max",
            ]
        )

        for generation, values in enumerate(
            zip(*histories)
        ):

            values = list(values)

            writer.writerow(
                [
                    generation,
                    min(values),
                    statistics.fmean(values),
                    max(values),
                ]
            )


# ============================================================
# Графики
# ============================================================

def save_convergence_plot(
    path,
    histories,
    method_name,
):

    minimum_values = []
    mean_values = []
    maximum_values = []

    for values in zip(*histories):

        values = list(values)

        minimum_values.append(
            min(values)
        )

        mean_values.append(
            statistics.fmean(values)
        )

        maximum_values.append(
            max(values)
        )

    generations = range(
        len(mean_values)
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        generations,
        mean_values,
        label="Среднее",
    )

    plt.fill_between(
        generations,
        minimum_values,
        maximum_values,
        alpha=0.25,
        label="Минимум - максимум",
    )

    plt.xlabel(
        "Поколение"
    )

    plt.ylabel(
        "Лучшее значение f(x)"
    )

    plt.title(
        f"Сходимость: {method_name}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=160,
    )

    plt.close()


def save_comparison_plot(
    path,
    results,
):

    labels = list(
        results.keys()
    )

    values = [
        results[label]
        for label in labels
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        values,
        tick_labels=labels,
    )

    plt.ylabel(
        "Лучшее значение f(x)"
    )

    plt.title(
        "Сравнение результатов"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=160,
    )

    plt.close()


# ============================================================
# Конфигурация
# ============================================================

def load_config(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    if config["variant"] != VARIANT:

        raise ValueError(
            f"Ожидается вариант {VARIANT}"
        )

    if config["dimension"] != DIMENSION:

        raise ValueError(
            f"Для варианта 19 d = {DIMENSION}"
        )

    if config["independent_runs"] < 20:

        raise ValueError(
            "По заданию требуется минимум 20 запусков"
        )

    return config


# ============================================================
# Сохранение summary.txt
# ============================================================

def save_summary(
    path,
    results,
    evaluation_budget,
    runs,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Лабораторная работа №1\n"
        )

        file.write(
            "Вариант 19 — HappyCat\n\n"
        )

        file.write(
            f"Размерность: {DIMENSION}\n"
        )

        file.write(
            f"Область: [{LOWER_BOUND}, {UPPER_BOUND}]^{DIMENSION}\n"
        )

        file.write(
            f"Количество запусков: {runs}\n"
        )

        file.write(
            f"Бюджет вычислений на запуск: {evaluation_budget}\n"
        )

        file.write(
            f"Теоретический минимум: {THEORETICAL_MINIMUM}\n"
        )

        file.write(
            "Точка минимума: "
            + str(THEORETICAL_MINIMIZER)
            + "\n\n"
        )

        for method_name, values in results.items():

            stats = calculate_statistics(
                values
            )

            file.write(
                method_name + "\n"
            )

            file.write(
                f"best   = {stats['best']:.12f}\n"
            )

            file.write(
                f"mean   = {stats['mean']:.12f}\n"
            )

            file.write(
                f"median = {stats['median']:.12f}\n"
            )

            file.write(
                f"std    = {stats['std']:.12f}\n"
            )

            file.write(
                f"worst  = {stats['worst']:.12f}\n\n"
            )


# ============================================================
# Полный эксперимент
# ============================================================

def run_experiment(
    config,
    output_dir,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = config[
        "independent_runs"
    ]

    base_seed = config[
        "base_seed"
    ]

    ga_data = config[
        "ga"
    ]

    probabilities = config[
        "compared_mutation_probabilities"
    ]

    if len(probabilities) != 2:
        raise ValueError(
            "Нужно указать две вероятности мутации"
        )

    base_ga_config = GAConfig(
        population_size=ga_data[
            "population_size"
        ],
        generations=ga_data[
            "generations"
        ],
        crossover_probability=ga_data[
            "crossover_probability"
        ],
        mutation_probability=probabilities[0],
        mutation_sigma=ga_data[
            "mutation_sigma"
        ],
        tournament_size=ga_data[
            "tournament_size"
        ],
        elitism=ga_data[
            "elitism"
        ],
    )

    if base_ga_config.population_size < 30:

        raise ValueError(
            "Размер популяции должен быть >= 30"
        )

    rows = []

    all_results = {}

    # ========================================================
    # Две конфигурации ГА
    # Отличается только вероятность мутации
    # ========================================================

    for mutation_probability in probabilities:

        current_config = replace(
            base_ga_config,
            mutation_probability=mutation_probability,
        )

        method_name = (
            f"GA_mut_{mutation_probability:.2f}"
        )

        values = []
        histories = []

        print()
        print(
            "=" * 60
        )

        print(
            f"Запуск серии: {method_name}"
        )

        print(
            "=" * 60
        )

        for run_index in range(runs):

            seed = (
                base_seed
                + run_index
            )

            (
                best_fitness,
                best_vector,
                history,
            ) = run_ga(
                current_config,
                seed,
            )

            values.append(
                best_fitness
            )

            histories.append(
                history
            )

            rows.append(
                {
                    "method": method_name,
                    "run": run_index + 1,
                    "seed": seed,
                    "best_fitness": best_fitness,
                    "best_vector": " ".join(
                        f"{value:.10f}"
                        for value in best_vector
                    ),
                    "evaluations":
                        current_config.evaluation_budget,
                }
            )

            print(
                f"Run {run_index + 1:02d} | "
                f"seed={seed} | "
                f"best={best_fitness:.10f}"
            )

        all_results[
            method_name
        ] = values

        save_convergence_csv(
            output_dir
            / f"convergence_{method_name}.csv",
            histories,
        )

        save_convergence_plot(
            output_dir
            / f"convergence_{method_name}.png",
            histories,
            method_name,
        )

    # ========================================================
    # Random Search
    # ========================================================

    evaluation_budget = (
        base_ga_config.evaluation_budget
    )

    random_values = []

    print()
    print(
        "=" * 60
    )

    print(
        "Запуск серии: RandomSearch"
    )

    print(
        "=" * 60
    )

    for run_index in range(runs):

        seed = (
            base_seed
            + run_index
        )

        best_fitness, best_vector = (
            random_search(
                evaluation_budget,
                seed,
            )
        )

        random_values.append(
            best_fitness
        )

        rows.append(
            {
                "method": "RandomSearch",
                "run": run_index + 1,
                "seed": seed,
                "best_fitness": best_fitness,
                "best_vector": " ".join(
                    f"{value:.10f}"
                    for value in best_vector
                ),
                "evaluations":
                    evaluation_budget,
            }
        )

        print(
            f"Run {run_index + 1:02d} | "
            f"seed={seed} | "
            f"best={best_fitness:.10f}"
        )

    all_results[
        "RandomSearch"
    ] = random_values

    # ========================================================
    # Итоговые файлы
    # ========================================================

    save_runs_csv(
        output_dir / "runs.csv",
        rows,
    )

    save_summary(
        output_dir / "summary.txt",
        all_results,
        evaluation_budget,
        runs,
    )

    save_comparison_plot(
        output_dir / "comparison.png",
        all_results,
    )

    # ========================================================
    # Вывод статистики в консоль
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "ИТОГОВАЯ СТАТИСТИКА"
    )

    print(
        "=" * 60
    )

    for method_name, values in all_results.items():

        stats = calculate_statistics(
            values
        )

        print()
        print(
            method_name
        )

        print(
            f"Best:   {stats['best']:.10f}"
        )

        print(
            f"Mean:   {stats['mean']:.10f}"
        )

        print(
            f"Median: {stats['median']:.10f}"
        )

        print(
            f"Std:    {stats['std']:.10f}"
        )

        print(
            f"Worst:  {stats['worst']:.10f}"
        )

    print()
    print(
        f"Результаты сохранены в: "
        f"{output_dir.resolve()}"
    )


# ============================================================
# main
# ============================================================

def main():

    current_dir = Path(
        __file__
    ).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Лабораторная работа №1, "
            "вариант 19 — HappyCat"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=current_dir / "config.json",
        help="Путь к config.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=current_dir / "results",
        help="Папка для результатов",
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    run_experiment(
        config,
        args.output,
    )


if __name__ == "__main__":
    main()