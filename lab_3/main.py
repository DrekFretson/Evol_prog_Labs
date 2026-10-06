from __future__ import annotations

import argparse
import csv
import json
import statistics

from pathlib import Path
from typing import (
    List,
    Sequence,
    Tuple,
)

import matplotlib.pyplot as plt

from nsga2 import (
    Individual,
    crowding_distance,
    non_dominated_sort,
    run_nsga2,
    run_weighted_ga,
    weighted_score,
)

from traffic_sim import (
    generate_scenario,
    objectives,
)


VARIANT = 19

BASE_DIR = Path(
    __file__
).resolve().parent

RESULTS_DIR = (
    BASE_DIR
    / "results"
)


# ============================================================
# Конфигурация
# ============================================================

def load_config(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )

    if (
        config[
            "variant"
        ]
        != VARIANT
    ):
        raise ValueError(
            "Ожидается вариант 19."
        )

    if (
        config[
            "track"
        ]
        != "A"
    ):
        raise ValueError(
            "Эта реализация "
            "предназначена "
            "для трека A."
        )

    if (
        config[
            "independent_runs"
        ]
        < 20
    ):
        raise ValueError(
            "Требуется минимум "
            "20 независимых запусков."
        )

    if (
        config[
            "nsga2"
        ][
            "population_size"
        ]
        < 30
    ):
        raise ValueError(
            "Размер популяции "
            "должен быть минимум 30."
        )

    if (
        config[
            "nsga2"
        ][
            "min_green"
        ]
        >=
        config[
            "nsga2"
        ][
            "max_green"
        ]
    ):
        raise ValueError(
            "min_green должен быть "
            "меньше max_green."
        )

    weights = (
        config[
            "weighted_sum"
        ][
            "weights"
        ]
    )

    scales = (
        config[
            "weighted_sum"
        ][
            "normalization_scales"
        ]
    )

    if (
        len(weights) != 3
        or len(scales) != 3
    ):
        raise ValueError(
            "Для трёх критериев "
            "необходимо три веса "
            "и три масштаба."
        )

    return config


# ============================================================
# Удаление дублей
# ============================================================

def unique_individuals(
    individuals: Sequence[
        Individual
    ],
) -> List[
    Individual
]:

    result = {}

    for individual in individuals:

        result[
            tuple(
                individual.genes
            )
        ] = individual.clone()

    return list(
        result.values()
    )


# ============================================================
# Общий фронт Парето
# ============================================================

def global_pareto_front(
    individuals,
):

    unique = (
        unique_individuals(
            individuals
        )
    )

    if not unique:
        return []

    front = (
        non_dominated_sort(
            unique
        )[0]
    )

    crowding_distance(
        front
    )

    return sorted(
        [
            individual.clone()
            for individual
            in front
        ],
        key=lambda individual: (
            individual.objectives[
                0
            ],
            individual.objectives[
                1
            ],
            individual.objectives[
                2
            ],
        ),
    )


# ============================================================
# Три характерных решения
# ============================================================

def select_representatives(
    front,
):

    if not front:
        return []

    selected = []
    used = set()

    criteria = [
        (
            "Минимальная задержка",
            0,
        ),
        (
            "Минимальная очередь",
            1,
        ),
        (
            "Минимум остановок",
            2,
        ),
    ]

    for (
        label,
        objective_index,
    ) in criteria:

        ordered = sorted(
            front,
            key=lambda individual:
            individual.objectives[
                objective_index
            ],
        )

        for individual in ordered:

            key = tuple(
                individual.genes
            )

            if key in used:
                continue

            selected.append(
                (
                    label,
                    individual.clone(),
                )
            )

            used.add(
                key
            )

            break

    # На случай, если фронт маленький
    # или несколько экстремумов совпали.

    for individual in front:

        if (
            len(selected)
            >= 3
        ):
            break

        key = tuple(
            individual.genes
        )

        if key in used:
            continue

        selected.append(
            (
                "Компромиссное решение",
                individual.clone(),
            )
        )

        used.add(
            key
        )

    return selected


# ============================================================
# CSV
# ============================================================

def save_csv(
    path,
    header,
    rows,
):

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            header
        )

        writer.writerows(
            rows
        )


def save_front(
    front,
    config,
):

    transition = int(
        config[
            "simulation"
        ][
            "transition_seconds"
        ]
    )

    rows = []

    for individual in front:

        green_ns = (
            individual.genes[
                0
            ]
        )

        green_ew = (
            individual.genes[
                1
            ]
        )

        cycle = (
            green_ns
            + green_ew
            + 2 * transition
        )

        rows.append(
            [
                green_ns,
                green_ew,
                cycle,
                individual.objectives[
                    0
                ],
                individual.objectives[
                    1
                ],
                individual.objectives[
                    2
                ],
                individual.crowding,
            ]
        )

    save_csv(
        RESULTS_DIR
        / "pareto_front.csv",
        [
            "green_ns",
            "green_ew",
            "cycle_seconds",
            "average_delay",
            "max_queue",
            "stops",
            "crowding_distance",
        ],
        rows,
    )


def save_representatives(
    representatives,
    config,
):

    transition = int(
        config[
            "simulation"
        ][
            "transition_seconds"
        ]
    )

    rows = []

    for (
        label,
        individual,
    ) in representatives:

        green_ns = (
            individual.genes[
                0
            ]
        )

        green_ew = (
            individual.genes[
                1
            ]
        )

        rows.append(
            [
                label,
                green_ns,
                green_ew,
                green_ns
                + green_ew
                + 2 * transition,
                individual.objectives[
                    0
                ],
                individual.objectives[
                    1
                ],
                individual.objectives[
                    2
                ],
            ]
        )

    save_csv(
        RESULTS_DIR
        / "representative_solutions.csv",
        [
            "label",
            "green_ns",
            "green_ew",
            "cycle_seconds",
            "average_delay",
            "max_queue",
            "stops",
        ],
        rows,
    )


# ============================================================
# График фронта Парето
# ============================================================

def plot_pareto(
    front,
    representatives,
    weighted_best,
    x_index,
    y_index,
    x_label,
    y_label,
    filename,
):

    plt.figure(
        figsize=(
            9,
            6,
        )
    )

    plt.scatter(
        [
            individual.objectives[
                x_index
            ]
            for individual
            in front
        ],
        [
            individual.objectives[
                y_index
            ]
            for individual
            in front
        ],
        alpha=0.75,
        label=(
            "NSGA-II Pareto"
        ),
    )

    for (
        label,
        individual,
    ) in representatives:

        plt.scatter(
            [
                individual.objectives[
                    x_index
                ]
            ],
            [
                individual.objectives[
                    y_index
                ]
            ],
            marker="x",
            s=100,
            label=label,
        )

    plt.scatter(
        [
            weighted_best.objectives[
                x_index
            ]
        ],
        [
            weighted_best.objectives[
                y_index
            ]
        ],
        marker="D",
        s=70,
        label="Weighted sum",
    )

    plt.xlabel(
        x_label
    )

    plt.ylabel(
        y_label
    )

    plt.title(
        "Проекция фронта Парето"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / filename,
        dpi=160,
    )

    plt.close()


# ============================================================
# Графики сходимости
# ============================================================

def plot_convergence(
    histories,
):

    labels = [
        "Минимальная задержка",
        "Минимальная "
        "максимальная очередь",
        "Минимум остановок",
    ]

    filenames = [
        "convergence_delay.png",
        "convergence_queue.png",
        "convergence_stops.png",
    ]

    generation_count = len(
        histories[
            0
        ]
    )

    for objective_index in range(
        3
    ):

        means = []

        for generation in range(
            generation_count
        ):

            values = [
                history[
                    generation
                ][
                    objective_index
                ]
                for history
                in histories
            ]

            means.append(
                statistics.fmean(
                    values
                )
            )

        plt.figure(
            figsize=(
                9,
                5,
            )
        )

        plt.plot(
            range(
                generation_count
            ),
            means,
            label=labels[
                objective_index
            ],
        )

        plt.xlabel(
            "Поколение"
        )

        plt.ylabel(
            labels[
                objective_index
            ]
        )

        plt.title(
            "Сходимость NSGA-II "
            "по 20 запускам"
        )

        plt.grid(
            alpha=0.25
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR
            / filenames[
                objective_index
            ],
            dpi=160,
        )

        plt.close()


# ============================================================
# Статистика
# ============================================================

def descriptive_stats(
    values,
):

    return {
        "best": min(
            values
        ),

        "mean":
        statistics.fmean(
            values
        ),

        "median":
        statistics.median(
            values
        ),

        "std":
        statistics.stdev(
            values
        )
        if len(values) > 1
        else 0.0,

        "worst": max(
            values
        ),
    }


# ============================================================
# Проверка доминирования weighted-sum решения
# ============================================================

def is_dominated_by_front(
    individual,
    front,
):

    for candidate in front:

        no_worse = all(
            a <= b
            for a, b in zip(
                candidate.objectives,
                individual.objectives,
            )
        )

        strictly_better = any(
            a < b
            for a, b in zip(
                candidate.objectives,
                individual.objectives,
            )
        )

        if (
            no_worse
            and strictly_better
        ):
            return True

    return False


# ============================================================
# Эксперименты
# ============================================================

def run_experiments(
    config,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Один фиксированный сценарий движения.
    # Все алгоритмы сравниваются
    # на одинаковых входных данных.
    # --------------------------------------------------------

    scenario = (
        generate_scenario(
            config,
            int(
                config[
                    "scenario_seed"
                ]
            ),
        )
    )

    def evaluator(
        genes,
    ):
        return objectives(
            genes,
            scenario,
            config,
        )

    runs = int(
        config[
            "independent_runs"
        ]
    )

    base_seed = int(
        config[
            "base_seed"
        ]
    )

    print(
        "Сценарий движения:"
    )

    print(
        "  горизонт: "
        f"{config['simulation']['horizon_seconds']} с"
    )

    print(
        "  всего прибытий: "
        f"{scenario.total_arrivals}"
    )

    # ========================================================
    # NSGA-II
    # ========================================================

    all_fronts = []

    nsga_histories = []

    nsga_rows = []

    print()

    print(
        "=" * 72
    )

    print(
        "NSGA-II"
    )

    print(
        "=" * 72
    )

    for run_index in range(
        runs
    ):

        seed = (
            base_seed
            + run_index
        )

        (
            front,
            history,
            evaluations,
        ) = run_nsga2(
            config,
            evaluator,
            seed,
        )

        all_fronts.extend(
            front
        )

        nsga_histories.append(
            history
        )

        best_delay = min(
            individual.objectives[
                0
            ]
            for individual
            in front
        )

        best_queue = min(
            individual.objectives[
                1
            ]
            for individual
            in front
        )

        best_stops = min(
            individual.objectives[
                2
            ]
            for individual
            in front
        )

        nsga_rows.append(
            [
                run_index + 1,
                seed,
                len(front),
                evaluations,
                best_delay,
                best_queue,
                best_stops,
            ]
        )

        print(
            f"Run "
            f"{run_index + 1:02d} | "
            f"seed={seed} | "
            f"front={len(front):02d} | "
            f"delay={best_delay:.4f} | "
            f"queue={best_queue:.0f} | "
            f"stops={best_stops:.0f}"
        )

    global_front = (
        global_pareto_front(
            all_fronts
        )
    )

    representatives = (
        select_representatives(
            global_front
        )
    )

    # ========================================================
    # Weighted sum
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "Weighted-sum GA"
    )

    print(
        "=" * 72
    )

    weighted_rows = []

    weighted_solutions = []

    for run_index in range(
        runs
    ):

        seed = (
            base_seed
            + run_index
        )

        (
            best,
            _,
            evaluations,
        ) = run_weighted_ga(
            config,
            evaluator,
            seed,
        )

        score = (
            weighted_score(
                best.objectives,
                config,
            )
        )

        weighted_solutions.append(
            best
        )

        weighted_rows.append(
            [
                run_index + 1,
                seed,
                best.genes[
                    0
                ],
                best.genes[
                    1
                ],
                best.objectives[
                    0
                ],
                best.objectives[
                    1
                ],
                best.objectives[
                    2
                ],
                score,
                evaluations,
            ]
        )

        print(
            f"Run "
            f"{run_index + 1:02d} | "
            f"seed={seed} | "
            f"genes={best.genes} | "
            f"delay="
            f"{best.objectives[0]:.4f} | "
            f"queue="
            f"{best.objectives[1]:.0f} | "
            f"stops="
            f"{best.objectives[2]:.0f} | "
            f"score={score:.5f}"
        )

    weighted_best = min(
        weighted_solutions,
        key=lambda individual:
        weighted_score(
            individual.objectives,
            config,
        ),
    )

    # ========================================================
    # CSV
    # ========================================================

    save_csv(
        RESULTS_DIR
        / "nsga_runs.csv",
        [
            "run",
            "seed",
            "front_size",
            "evaluations",
            "best_delay",
            "best_queue",
            "best_stops",
        ],
        nsga_rows,
    )

    save_csv(
        RESULTS_DIR
        / "weighted_sum_runs.csv",
        [
            "run",
            "seed",
            "green_ns",
            "green_ew",
            "average_delay",
            "max_queue",
            "stops",
            "weighted_score",
            "evaluations",
        ],
        weighted_rows,
    )

    save_front(
        global_front,
        config,
    )

    save_representatives(
        representatives,
        config,
    )

    # ========================================================
    # Проекции фронта Парето
    # ========================================================

    plot_pareto(
        global_front,
        representatives,
        weighted_best,
        0,
        1,
        "Средняя задержка, с",
        "Максимальная длина очереди",
        "pareto_delay_queue.png",
    )

    plot_pareto(
        global_front,
        representatives,
        weighted_best,
        0,
        2,
        "Средняя задержка, с",
        "Число остановок",
        "pareto_delay_stops.png",
    )

    plot_pareto(
        global_front,
        representatives,
        weighted_best,
        1,
        2,
        "Максимальная длина очереди",
        "Число остановок",
        "pareto_queue_stops.png",
    )

    plot_convergence(
        nsga_histories
    )

    # ========================================================
    # Итоговая статистика
    # ========================================================

    front_sizes = [
        row[
            2
        ]
        for row
        in nsga_rows
    ]

    delay_stats = (
        descriptive_stats(
            [
                row[
                    4
                ]
                for row
                in nsga_rows
            ]
        )
    )

    queue_stats = (
        descriptive_stats(
            [
                row[
                    5
                ]
                for row
                in nsga_rows
            ]
        )
    )

    stops_stats = (
        descriptive_stats(
            [
                row[
                    6
                ]
                for row
                in nsga_rows
            ]
        )
    )

    weighted_scores = [
        weighted_score(
            individual.objectives,
            config,
        )
        for individual
        in weighted_solutions
    ]

    weighted_stats = (
        descriptive_stats(
            weighted_scores
        )
    )

    dominated_weighted = sum(
        1
        for individual
        in weighted_solutions
        if is_dominated_by_front(
            individual,
            global_front,
        )
    )

    lines = [
        "Лабораторная работа №3",

        (
            "Вариант 19, трек A — "
            "управление светофором "
            "в симуляции"
        ),

        "",

        (
            "Критерии: минимум "
            "средней задержки, "
            "максимальной длины "
            "очереди и числа остановок."
        ),

        (
            "Независимых запусков: "
            f"{runs}"
        ),

        (
            "Размер популяции: "
            f"{config['nsga2']['population_size']}"
        ),

        (
            "Поколений: "
            f"{config['nsga2']['generations']}"
        ),

        (
            "Оценок на один запуск: "
            f"{nsga_rows[0][3]}"
        ),

        (
            "Глобальный размер "
            "фронта Парето: "
            f"{len(global_front)}"
        ),

        "",

        (
            "NSGA-II: размер "
            "фронта по запускам"
        ),

        (
            f"  min:    "
            f"{min(front_sizes)}"
        ),

        (
            f"  mean:   "
            f"{statistics.fmean(front_sizes):.3f}"
        ),

        (
            f"  median: "
            f"{statistics.median(front_sizes):.3f}"
        ),

        (
            f"  max:    "
            f"{max(front_sizes)}"
        ),

        "",
    ]

    for (
        title,
        values,
    ) in [
        (
            "Лучший delay "
            "по запускам",
            delay_stats,
        ),
        (
            "Лучшая максимальная "
            "очередь по запускам",
            queue_stats,
        ),
        (
            "Лучшее число "
            "остановок по запускам",
            stops_stats,
        ),
    ]:

        lines.extend(
            [
                title,

                (
                    f"  best:   "
                    f"{values['best']:.6f}"
                ),

                (
                    f"  mean:   "
                    f"{values['mean']:.6f}"
                ),

                (
                    f"  median: "
                    f"{values['median']:.6f}"
                ),

                (
                    f"  std:    "
                    f"{values['std']:.6f}"
                ),

                (
                    f"  worst:  "
                    f"{values['worst']:.6f}"
                ),

                "",
            ]
        )

    lines.extend(
        [
            "Weighted-sum GA",

            (
                "  weighted score best:   "
                f"{weighted_stats['best']:.6f}"
            ),

            (
                "  weighted score mean:   "
                f"{weighted_stats['mean']:.6f}"
            ),

            (
                "  weighted score median: "
                f"{weighted_stats['median']:.6f}"
            ),

            (
                "  weighted score std:    "
                f"{weighted_stats['std']:.6f}"
            ),

            (
                "  weighted score worst:  "
                f"{weighted_stats['worst']:.6f}"
            ),

            (
                "  доминируемых "
                "глобальным фронтом "
                "NSGA-II: "
                f"{dominated_weighted}/{runs}"
            ),

            "",

            "Лучшее weighted-sum решение",

            (
                "  green_ns: "
                f"{weighted_best.genes[0]}"
            ),

            (
                "  green_ew: "
                f"{weighted_best.genes[1]}"
            ),

            (
                "  delay:    "
                f"{weighted_best.objectives[0]:.6f}"
            ),

            (
                "  queue:    "
                f"{weighted_best.objectives[1]:.6f}"
            ),

            (
                "  stops:    "
                f"{weighted_best.objectives[2]:.0f}"
            ),

            "",

            (
                "Характерные решения "
                "фронта Парето"
            ),
        ]
    )

    for (
        label,
        individual,
    ) in representatives:

        lines.extend(
            [
                label,

                (
                    "  green_ns="
                    f"{individual.genes[0]}, "
                    "green_ew="
                    f"{individual.genes[1]}"
                ),

                (
                    "  delay="
                    f"{individual.objectives[0]:.6f}, "
                    "queue="
                    f"{individual.objectives[1]:.6f}, "
                    "stops="
                    f"{individual.objectives[2]:.0f}"
                ),
            ]
        )

    (
        RESULTS_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Консоль
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "ИТОГОВАЯ СТАТИСТИКА"
    )

    print(
        "=" * 72
    )

    print(
        "\n".join(
            lines
        )
    )

    print()

    print(
        "Результаты сохранены в: "
        f"{RESULTS_DIR}"
    )


# ============================================================
# main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ЛР №3, вариант 19, "
            "трек A: NSGA-II "
            "для управления светофором"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            BASE_DIR
            / "config.json"
        ),
        help=(
            "Путь к config.json"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    run_experiments(
        config
    )


if __name__ == "__main__":
    main()