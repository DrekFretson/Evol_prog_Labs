from __future__ import annotations

import math
import random

from dataclasses import dataclass
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
)


Objectives = Tuple[
    float,
    float,
    float,
]

Evaluator = Callable[
    [Sequence[int]],
    Objectives,
]


@dataclass
class Individual:
    genes: List[int]

    objectives: Optional[
        Objectives
    ] = None

    rank: int = 0

    crowding: float = 0.0

    def clone(self):
        return Individual(
            genes=self.genes[:],
            objectives=self.objectives,
            rank=self.rank,
            crowding=self.crowding,
        )


# ============================================================
# Парето-доминирование
# ============================================================

def dominates(
    a: Individual,
    b: Individual,
) -> bool:

    if (
        a.objectives is None
        or b.objectives is None
    ):
        raise ValueError(
            "Особи должны быть оценены."
        )

    no_worse = all(
        x <= y
        for x, y
        in zip(
            a.objectives,
            b.objectives,
        )
    )

    strictly_better = any(
        x < y
        for x, y
        in zip(
            a.objectives,
            b.objectives,
        )
    )

    return (
        no_worse
        and strictly_better
    )


# ============================================================
# Недоминируемая сортировка
# ============================================================

def non_dominated_sort(
    population: Sequence[
        Individual
    ],
):
    dominates_map = {
        id(individual): []
        for individual
        in population
    }

    dominated_count = {
        id(individual): 0
        for individual
        in population
    }

    first_front = []

    for p in population:

        for q in population:

            if p is q:
                continue

            if dominates(
                p,
                q,
            ):
                dominates_map[
                    id(p)
                ].append(
                    q
                )

            elif dominates(
                q,
                p,
            ):
                dominated_count[
                    id(p)
                ] += 1

        if (
            dominated_count[
                id(p)
            ]
            == 0
        ):
            p.rank = 0

            first_front.append(
                p
            )

    fronts = [
        first_front
    ]

    rank = 0

    while fronts[
        rank
    ]:

        next_front = []

        for p in fronts[
            rank
        ]:

            for q in (
                dominates_map[
                    id(p)
                ]
            ):

                dominated_count[
                    id(q)
                ] -= 1

                if (
                    dominated_count[
                        id(q)
                    ]
                    == 0
                ):
                    q.rank = (
                        rank + 1
                    )

                    next_front.append(
                        q
                    )

        rank += 1

        fronts.append(
            next_front
        )

    fronts.pop()

    return fronts


# ============================================================
# Crowding distance
# ============================================================

def crowding_distance(
    front: Sequence[
        Individual
    ],
):
    if not front:
        return

    for individual in front:
        individual.crowding = (
            0.0
        )

    if len(front) <= 2:

        for individual in front:
            individual.crowding = (
                math.inf
            )

        return

    # У нас три критерия.

    for objective_index in range(
        3
    ):
        ordered = sorted(
            front,
            key=lambda x:
            x.objectives[
                objective_index
            ],
        )

        ordered[
            0
        ].crowding = math.inf

        ordered[
            -1
        ].crowding = math.inf

        minimum = (
            ordered[
                0
            ].objectives[
                objective_index
            ]
        )

        maximum = (
            ordered[
                -1
            ].objectives[
                objective_index
            ]
        )

        if maximum == minimum:
            continue

        scale = (
            maximum
            - minimum
        )

        for i in range(
            1,
            len(ordered) - 1,
        ):

            if math.isinf(
                ordered[
                    i
                ].crowding
            ):
                continue

            previous_value = (
                ordered[
                    i - 1
                ].objectives[
                    objective_index
                ]
            )

            next_value = (
                ordered[
                    i + 1
                ].objectives[
                    objective_index
                ]
            )

            ordered[
                i
            ].crowding += (
                next_value
                - previous_value
            ) / scale


def prepare(
    population,
):
    fronts = (
        non_dominated_sort(
            population
        )
    )

    for front in fronts:
        crowding_distance(
            front
        )

    return fronts


# ============================================================
# Оценивание
# ============================================================

def evaluate(
    population,
    evaluator,
):
    count = 0

    for individual in population:

        if (
            individual.objectives
            is None
        ):
            individual.objectives = (
                evaluator(
                    individual.genes
                )
            )

            count += 1

    return count


# ============================================================
# Начальная популяция
# ============================================================

def random_population(
    config,
    rng,
):
    cfg = config[
        "nsga2"
    ]

    population = []

    for _ in range(
        int(
            cfg[
                "population_size"
            ]
        )
    ):

        genes = [
            rng.randint(
                int(
                    cfg[
                        "min_green"
                    ]
                ),
                int(
                    cfg[
                        "max_green"
                    ]
                ),
            ),
            rng.randint(
                int(
                    cfg[
                        "min_green"
                    ]
                ),
                int(
                    cfg[
                        "max_green"
                    ]
                ),
            ),
        ]

        population.append(
            Individual(
                genes
            )
        )

    return population


# ============================================================
# Турнир NSGA-II
# ============================================================

def tournament_nsga(
    population,
    rng,
):
    a, b = rng.sample(
        list(
            population
        ),
        2,
    )

    if a.rank != b.rank:

        if (
            a.rank
            < b.rank
        ):
            return a

        return b

    if (
        a.crowding
        != b.crowding
    ):

        if (
            a.crowding
            > b.crowding
        ):
            return a

        return b

    return a


# ============================================================
# Ограничение генов
# ============================================================

def clamp(
    value,
    minimum,
    maximum,
):
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


# ============================================================
# Кроссовер
# ============================================================

def crossover(
    parent1,
    parent2,
    config,
    rng,
):
    cfg = config[
        "nsga2"
    ]

    minimum = int(
        cfg[
            "min_green"
        ]
    )

    maximum = int(
        cfg[
            "max_green"
        ]
    )

    child1 = []
    child2 = []

    for gene1, gene2 in zip(
        parent1.genes,
        parent2.genes,
    ):
        alpha = rng.uniform(
            -0.25,
            1.25,
        )

        value1 = round(
            alpha * gene1
            + (
                1.0 - alpha
            ) * gene2
        )

        value2 = round(
            alpha * gene2
            + (
                1.0 - alpha
            ) * gene1
        )

        child1.append(
            clamp(
                value1,
                minimum,
                maximum,
            )
        )

        child2.append(
            clamp(
                value2,
                minimum,
                maximum,
            )
        )

    return (
        Individual(
            child1
        ),
        Individual(
            child2
        ),
    )


# ============================================================
# Мутация
# ============================================================

def mutate(
    individual,
    config,
    rng,
):
    cfg = config[
        "nsga2"
    ]

    probability = float(
        cfg[
            "mutation_probability"
        ]
    )

    sigma = float(
        cfg[
            "mutation_sigma"
        ]
    )

    minimum = int(
        cfg[
            "min_green"
        ]
    )

    maximum = int(
        cfg[
            "max_green"
        ]
    )

    for i in range(
        len(
            individual.genes
        )
    ):

        if (
            rng.random()
            < probability
        ):

            value = round(
                individual.genes[
                    i
                ]
                + rng.gauss(
                    0.0,
                    sigma,
                )
            )

            individual.genes[
                i
            ] = clamp(
                value,
                minimum,
                maximum,
            )

    individual.objectives = (
        None
    )


# ============================================================
# Потомки
# ============================================================

def make_children(
    population,
    config,
    rng,
):
    cfg = config[
        "nsga2"
    ]

    size = int(
        cfg[
            "population_size"
        ]
    )

    crossover_probability = (
        float(
            cfg[
                "crossover_probability"
            ]
        )
    )

    children = []

    while (
        len(children)
        < size
    ):
        parent1 = (
            tournament_nsga(
                population,
                rng,
            )
        )

        parent2 = (
            tournament_nsga(
                population,
                rng,
            )
        )

        if (
            rng.random()
            < crossover_probability
        ):
            child1, child2 = (
                crossover(
                    parent1,
                    parent2,
                    config,
                    rng,
                )
            )

        else:
            child1 = (
                Individual(
                    parent1.genes[:]
                )
            )

            child2 = (
                Individual(
                    parent2.genes[:]
                )
            )

        mutate(
            child1,
            config,
            rng,
        )

        mutate(
            child2,
            config,
            rng,
        )

        children.append(
            child1
        )

        if (
            len(children)
            < size
        ):
            children.append(
                child2
            )

    return children


# ============================================================
# Отбор следующего поколения
# ============================================================

def environmental_selection(
    combined,
    size,
):
    new_population = []

    fronts = (
        non_dominated_sort(
            combined
        )
    )

    for front in fronts:

        crowding_distance(
            front
        )

        if (
            len(
                new_population
            )
            + len(front)
            <= size
        ):
            new_population.extend(
                front
            )

        else:
            front = sorted(
                front,
                key=lambda x:
                x.crowding,
                reverse=True,
            )

            remaining = (
                size
                - len(
                    new_population
                )
            )

            new_population.extend(
                front[
                    :remaining
                ]
            )

            break

    result = [
        individual.clone()
        for individual
        in new_population
    ]

    prepare(
        result
    )

    return result


def minima(
    population,
):
    return tuple(
        min(
            individual.objectives[
                objective_index
            ]
            for individual
            in population
        )
        for objective_index
        in range(3)
    )


# ============================================================
# Полный NSGA-II
# ============================================================

def run_nsga2(
    config,
    evaluator,
    seed,
):
    rng = random.Random(
        seed
    )

    population = (
        random_population(
            config,
            rng,
        )
    )

    evaluations = (
        evaluate(
            population,
            evaluator,
        )
    )

    prepare(
        population
    )

    history = [
        minima(
            population
        )
    ]

    size = int(
        config[
            "nsga2"
        ][
            "population_size"
        ]
    )

    generations = int(
        config[
            "nsga2"
        ][
            "generations"
        ]
    )

    for _ in range(
        generations
    ):
        children = (
            make_children(
                population,
                config,
                rng,
            )
        )

        evaluations += (
            evaluate(
                children,
                evaluator,
            )
        )

        population = (
            environmental_selection(
                population
                + children,
                size,
            )
        )

        history.append(
            minima(
                population
            )
        )

    front = (
        non_dominated_sort(
            population
        )[0]
    )

    crowding_distance(
        front
    )

    return (
        [
            individual.clone()
            for individual
            in front
        ],
        history,
        evaluations,
    )


# ============================================================
# Weighted sum
# ============================================================

def weighted_score(
    objectives,
    config,
):
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

    return sum(
        float(weight)
        * value
        / float(scale)
        for (
            weight,
            value,
            scale,
        )
        in zip(
            weights,
            objectives,
            scales,
        )
    )


def tournament_weighted(
    population,
    config,
    rng,
):
    size = int(
        config[
            "weighted_sum"
        ][
            "tournament_size"
        ]
    )

    candidates = rng.sample(
        list(
            population
        ),
        size,
    )

    return min(
        candidates,
        key=lambda individual:
        weighted_score(
            individual.objectives,
            config,
        ),
    )


# ============================================================
# ГА со взвешенной суммой
# ============================================================

def run_weighted_ga(
    config,
    evaluator,
    seed,
):
    rng = random.Random(
        seed
    )

    population = (
        random_population(
            config,
            rng,
        )
    )

    evaluations = (
        evaluate(
            population,
            evaluator,
        )
    )

    generations = int(
        config[
            "nsga2"
        ][
            "generations"
        ]
    )

    population_size = int(
        config[
            "nsga2"
        ][
            "population_size"
        ]
    )

    elite_count = int(
        config[
            "weighted_sum"
        ][
            "elite_count"
        ]
    )

    crossover_probability = (
        float(
            config[
                "nsga2"
            ][
                "crossover_probability"
            ]
        )
    )

    history = []

    for generation in range(
        generations + 1
    ):

        best = min(
            population,
            key=lambda individual:
            weighted_score(
                individual.objectives,
                config,
            ),
        )

        history.append(
            weighted_score(
                best.objectives,
                config,
            )
        )

        if (
            generation
            == generations
        ):
            break

        ordered = sorted(
            population,
            key=lambda individual:
            weighted_score(
                individual.objectives,
                config,
            ),
        )

        new_population = [
            Individual(
                ordered[
                    i
                ].genes[:],
                ordered[
                    i
                ].objectives,
            )
            for i in range(
                elite_count
            )
        ]

        while (
            len(
                new_population
            )
            < population_size
        ):
            parent1 = (
                tournament_weighted(
                    population,
                    config,
                    rng,
                )
            )

            parent2 = (
                tournament_weighted(
                    population,
                    config,
                    rng,
                )
            )

            if (
                rng.random()
                < crossover_probability
            ):
                child1, child2 = (
                    crossover(
                        parent1,
                        parent2,
                        config,
                        rng,
                    )
                )

            else:
                child1 = (
                    Individual(
                        parent1.genes[:]
                    )
                )

                child2 = (
                    Individual(
                        parent2.genes[:]
                    )
                )

            mutate(
                child1,
                config,
                rng,
            )

            mutate(
                child2,
                config,
                rng,
            )

            new_population.append(
                child1
            )

            if (
                len(
                    new_population
                )
                < population_size
            ):
                new_population.append(
                    child2
                )

        evaluations += (
            evaluate(
                new_population,
                evaluator,
            )
        )

        population = (
            new_population
        )

    best = min(
        population,
        key=lambda individual:
        weighted_score(
            individual.objectives,
            config,
        ),
    )

    return (
        best.clone(),
        history,
        evaluations,
    )