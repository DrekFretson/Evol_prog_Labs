from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from generate_data import generate_dataset


VARIANT = 19

BASE_DIR = Path(
    __file__
).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "data"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)


@dataclass(frozen=True)
class Exam:
    exam_id: int
    name: str


@dataclass(frozen=True)
class Evaluation:
    hard_conflict_pairs: int
    conflicting_students: int
    same_day_students: int
    used_slots: int
    soft_penalty: float
    score: float
    feasible: bool


@dataclass
class GARunResult:
    best_schedule: list[int]
    best_evaluation: Evaluation
    history: list[float]


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

    if config["variant"] != VARIANT:
        raise ValueError(
            f"Ожидается вариант {VARIANT}."
        )

    if config["num_exams"] < 20:
        raise ValueError(
            "Для задачи расписания "
            "требуется минимум 20 событий."
        )

    if (
        config["independent_runs"]
        < 20
    ):
        raise ValueError(
            "По заданию требуется минимум "
            "20 независимых запусков."
        )

    if (
        config["elite_count"]
        >= config["population_size"]
    ):
        raise ValueError(
            "elite_count должен быть "
            "меньше population_size."
        )

    return config


# ============================================================
# Генерация данных
# ============================================================

def ensure_data(
    config: dict,
    regenerate: bool = False,
) -> None:

    required_files = [
        DATA_DIR / "exams.csv",
        DATA_DIR / "enrollments.csv",
        DATA_DIR / "conflicts.csv",
    ]

    if (
        regenerate
        or not all(
            path.exists()
            for path
            in required_files
        )
    ):
        generate_dataset(
            root=BASE_DIR,
            seed=config[
                "data_seed"
            ],
            num_exams=config[
                "num_exams"
            ],
            num_students=config[
                "num_students"
            ],
        )


# ============================================================
# Загрузка данных
# ============================================================

def load_data():

    exams = []

    with (
        DATA_DIR
        / "exams.csv"
    ).open(
        encoding="utf-8-sig"
    ) as file:

        for row in csv.DictReader(
            file
        ):
            exams.append(
                Exam(
                    exam_id=int(
                        row[
                            "exam_id"
                        ]
                    ),
                    name=row[
                        "exam_name"
                    ],
                )
            )

    exams.sort(
        key=lambda exam:
        exam.exam_id
    )

    expected_ids = list(
        range(
            1,
            len(exams) + 1,
        )
    )

    actual_ids = [
        exam.exam_id
        for exam in exams
    ]

    if (
        actual_ids
        != expected_ids
    ):
        raise ValueError(
            "exam_id должны идти "
            "подряд от 1 до N."
        )

    # Количество студентов
    # на каждом экзамене.
    student_counts = [
        0
        for _ in exams
    ]

    with (
        DATA_DIR
        / "enrollments.csv"
    ).open(
        encoding="utf-8-sig"
    ) as file:

        for row in csv.DictReader(
            file
        ):
            exam_index = (
                int(
                    row[
                        "exam_id"
                    ]
                )
                - 1
            )

            student_counts[
                exam_index
            ] += 1

    conflicts = []

    with (
        DATA_DIR
        / "conflicts.csv"
    ).open(
        encoding="utf-8-sig"
    ) as file:

        for row in csv.DictReader(
            file
        ):
            conflicts.append(
                (
                    int(
                        row[
                            "exam_a"
                        ]
                    ) - 1,

                    int(
                        row[
                            "exam_b"
                        ]
                    ) - 1,

                    int(
                        row[
                            "shared_students"
                        ]
                    ),
                )
            )

    if len(conflicts) < 10:
        raise ValueError(
            "Для варианта расписания "
            "нужно минимум 10 конфликтов."
        )

    return (
        exams,
        conflicts,
        student_counts,
    )


# ============================================================
# Граф конфликтов
# ============================================================

def build_adjacency(
    exam_count,
    conflicts,
):

    adjacency = [
        []
        for _ in range(
            exam_count
        )
    ]

    for (
        exam_a,
        exam_b,
        shared_students,
    ) in conflicts:

        adjacency[
            exam_a
        ].append(
            (
                exam_b,
                shared_students,
            )
        )

        adjacency[
            exam_b
        ].append(
            (
                exam_a,
                shared_students,
            )
        )

    return adjacency


# ============================================================
# Оценка расписания
# ============================================================

def evaluate_schedule(
    schedule,
    conflicts,
    config,
):

    hard_conflict_pairs = 0

    conflicting_students = 0

    same_day_students = 0

    sessions_per_day = (
        config[
            "sessions_per_day"
        ]
    )

    for (
        exam_a,
        exam_b,
        shared_students,
    ) in conflicts:

        slot_a = schedule[
            exam_a
        ]

        slot_b = schedule[
            exam_b
        ]

        # Жёсткое нарушение:
        # общий студент должен
        # одновременно сдавать
        # два экзамена.
        if slot_a == slot_b:

            hard_conflict_pairs += 1

            conflicting_students += (
                shared_students
            )

        # Мягкое нарушение:
        # два экзамена попали
        # студенту на один день.
        elif (
            slot_a
            // sessions_per_day
            ==
            slot_b
            // sessions_per_day
        ):
            same_day_students += (
                shared_students
            )

    used_slots = len(
        set(schedule)
    )

    soft_penalty = (
        config[
            "same_day_weight"
        ]
        * same_day_students
        +
        config[
            "used_slot_weight"
        ]
        * used_slots
    )

    score = (
        config[
            "hard_conflict_penalty"
        ]
        * conflicting_students
        +
        soft_penalty
    )

    return Evaluation(
        hard_conflict_pairs=(
            hard_conflict_pairs
        ),
        conflicting_students=(
            conflicting_students
        ),
        same_day_students=(
            same_day_students
        ),
        used_slots=used_slots,
        soft_penalty=(
            soft_penalty
        ),
        score=score,
        feasible=(
            hard_conflict_pairs
            == 0
        ),
    )


# ============================================================
# Построение допустимого решения
# ============================================================

def construct_feasible_schedule(
    adjacency,
    config,
    rng,
    noise=None,
):

    exam_count = len(
        adjacency
    )

    num_slots = config[
        "num_slots"
    ]

    sessions_per_day = (
        config[
            "sessions_per_day"
        ]
    )

    if noise is None:
        noise = config[
            "constructor_noise"
        ]

    # Экзамены с большим
    # количеством конфликтов
    # рассматриваются первыми.
    order = list(
        range(
            exam_count
        )
    )

    rng.shuffle(order)

    order.sort(
        key=lambda exam:
        len(
            adjacency[
                exam
            ]
        ),
        reverse=True,
    )

    schedule = [
        -1
        for _ in range(
            exam_count
        )
    ]

    slot_counts = [
        0
        for _ in range(
            num_slots
        )
    ]

    for exam in order:

        candidates = []

        for slot in range(
            num_slots
        ):

            hard_students = 0

            same_day_students = 0

            day = (
                slot
                // sessions_per_day
            )

            for (
                neighbour,
                shared_students,
            ) in adjacency[
                exam
            ]:

                neighbour_slot = (
                    schedule[
                        neighbour
                    ]
                )

                if neighbour_slot == -1:
                    continue

                if (
                    neighbour_slot
                    == slot
                ):
                    hard_students += (
                        shared_students
                    )

                elif (
                    neighbour_slot
                    // sessions_per_day
                    ==
                    day
                ):
                    same_day_students += (
                        shared_students
                    )

            # Если в этом слоте
            # будет жёсткий конфликт,
            # слот запрещён.
            if hard_students != 0:
                continue

            if (
                slot_counts[
                    slot
                ]
                == 0
            ):
                new_slot_penalty = (
                    config[
                        "used_slot_weight"
                    ]
                )
            else:
                new_slot_penalty = 0.0

            candidate_score = (
                config[
                    "same_day_weight"
                ]
                * same_day_students
                +
                new_slot_penalty
                +
                rng.random()
                * noise
            )

            candidates.append(
                (
                    candidate_score,
                    slot,
                )
            )

        if not candidates:
            return None

        (
            _,
            chosen_slot,
        ) = min(
            candidates
        )

        schedule[
            exam
        ] = chosen_slot

        slot_counts[
            chosen_slot
        ] += 1

    return schedule


# ============================================================
# Начальная популяция
# ============================================================

def create_initial_population(
    adjacency,
    conflicts,
    config,
    rng,
):

    population = []

    while (
        len(population)
        <
        config[
            "population_size"
        ]
    ):

        schedule = (
            construct_feasible_schedule(
                adjacency,
                config,
                rng,
            )
        )

        if schedule is None:
            continue

        evaluation = (
            evaluate_schedule(
                schedule,
                conflicts,
                config,
            )
        )

        if evaluation.feasible:
            population.append(
                schedule
            )

    return population


# ============================================================
# Repair
# ============================================================

def repair_schedule(
    schedule,
    adjacency,
    conflicts,
    config,
    rng,
):

    repaired = schedule[:]

    exam_count = len(
        repaired
    )

    num_slots = config[
        "num_slots"
    ]

    sessions_per_day = (
        config[
            "sessions_per_day"
        ]
    )

    slot_counts = [
        0
        for _ in range(
            num_slots
        )
    ]

    for slot in repaired:
        slot_counts[
            slot
        ] += 1

    # Повторяем переносы,
    # пока существуют конфликты.
    for _ in range(
        exam_count * 4
    ):

        conflict_weight = [
            0
            for _ in range(
                exam_count
            )
        ]

        total_conflicting_students = 0

        for (
            exam_a,
            exam_b,
            shared_students,
        ) in conflicts:

            if (
                repaired[
                    exam_a
                ]
                ==
                repaired[
                    exam_b
                ]
            ):
                conflict_weight[
                    exam_a
                ] += (
                    shared_students
                )

                conflict_weight[
                    exam_b
                ] += (
                    shared_students
                )

                total_conflicting_students += (
                    shared_students
                )

        if (
            total_conflicting_students
            == 0
        ):
            return repaired

        max_weight = max(
            conflict_weight
        )

        candidates = [
            exam
            for exam, weight
            in enumerate(
                conflict_weight
            )
            if weight
            == max_weight
        ]

        exam = rng.choice(
            candidates
        )

        old_slot = repaired[
            exam
        ]

        slot_counts[
            old_slot
        ] -= 1

        variants = []

        for slot in range(
            num_slots
        ):

            if slot == old_slot:
                continue

            hard_students = 0
            same_day_students = 0

            day = (
                slot
                // sessions_per_day
            )

            for (
                neighbour,
                shared_students,
            ) in adjacency[
                exam
            ]:

                neighbour_slot = (
                    repaired[
                        neighbour
                    ]
                )

                if (
                    neighbour_slot
                    == slot
                ):
                    hard_students += (
                        shared_students
                    )

                elif (
                    neighbour_slot
                    // sessions_per_day
                    ==
                    day
                ):
                    same_day_students += (
                        shared_students
                    )

            if (
                slot_counts[
                    slot
                ]
                == 0
            ):
                new_slot_penalty = (
                    config[
                        "used_slot_weight"
                    ]
                )
            else:
                new_slot_penalty = 0.0

            score = (
                config[
                    "hard_conflict_penalty"
                ]
                * hard_students
                +
                config[
                    "same_day_weight"
                ]
                * same_day_students
                +
                new_slot_penalty
                +
                rng.random()
                * 1e-6
            )

            variants.append(
                (
                    score,
                    slot,
                )
            )

        (
            _,
            new_slot,
        ) = min(
            variants
        )

        repaired[
            exam
        ] = new_slot

        slot_counts[
            new_slot
        ] += 1

    # Если локальный repair
    # не справился,
    # строим новое допустимое
    # расписание.
    fallback = (
        construct_feasible_schedule(
            adjacency,
            config,
            rng,
            noise=0.25,
        )
    )

    if fallback is not None:
        return fallback

    return repaired


# ============================================================
# Турнирная селекция
# ============================================================

def tournament_selection(
    population,
    evaluations,
    config,
    rng,
):

    indices = rng.sample(
        range(
            len(population)
        ),
        config[
            "tournament_size"
        ],
    )

    best_index = min(
        indices,
        key=lambda index:
        evaluations[
            index
        ].score,
    )

    return population[
        best_index
    ][:]


# ============================================================
# Равномерный crossover
# ============================================================

def uniform_crossover(
    parent1,
    parent2,
    rng,
):

    child1 = []
    child2 = []

    for (
        gene1,
        gene2,
    ) in zip(
        parent1,
        parent2,
    ):

        if rng.random() < 0.5:

            child1.append(
                gene1
            )

            child2.append(
                gene2
            )

        else:

            child1.append(
                gene2
            )

            child2.append(
                gene1
            )

    return (
        child1,
        child2,
    )


# ============================================================
# Одноточечный crossover
# ============================================================

def one_point_crossover(
    parent1,
    parent2,
    rng,
):

    point = rng.randrange(
        1,
        len(parent1),
    )

    child1 = (
        parent1[:point]
        +
        parent2[point:]
    )

    child2 = (
        parent2[:point]
        +
        parent1[point:]
    )

    return (
        child1,
        child2,
    )


# ============================================================
# Мутация
# ============================================================

def mutate_schedule(
    schedule,
    config,
    rng,
):

    mutated = schedule[:]

    num_slots = config[
        "num_slots"
    ]

    for exam in range(
        len(mutated)
    ):

        if (
            rng.random()
            <
            config[
                "mutation_rate"
            ]
        ):

            old_slot = (
                mutated[
                    exam
                ]
            )

            new_slot = (
                rng.randrange(
                    num_slots - 1
                )
            )

            # Гарантируем,
            # что слот действительно
            # изменится.
            if (
                new_slot
                >= old_slot
            ):
                new_slot += 1

            mutated[
                exam
            ] = new_slot

    return mutated


# ============================================================
# Один запуск ГА
# ============================================================

def run_ga(
    adjacency,
    conflicts,
    config,
    seed,
    constraint_method,
    crossover_kind,
):

    rng = random.Random(
        seed
    )

    population = (
        create_initial_population(
            adjacency,
            conflicts,
            config,
            rng,
        )
    )

    best_schedule = None
    best_evaluation = None

    history = []

    for generation in range(
        config[
            "generations"
        ]
        + 1
    ):

        evaluations = [
            evaluate_schedule(
                schedule,
                conflicts,
                config,
            )
            for schedule
            in population
        ]

        feasible_indices = [
            index
            for index, evaluation
            in enumerate(
                evaluations
            )
            if evaluation.feasible
        ]

        if feasible_indices:

            generation_best_index = min(
                feasible_indices,
                key=lambda index:
                evaluations[
                    index
                ].soft_penalty,
            )

            generation_best_eval = (
                evaluations[
                    generation_best_index
                ]
            )

            if (
                best_evaluation
                is None
                or
                generation_best_eval
                .soft_penalty
                <
                best_evaluation
                .soft_penalty
            ):

                best_schedule = (
                    population[
                        generation_best_index
                    ][:]
                )

                best_evaluation = (
                    generation_best_eval
                )

        if best_evaluation is None:
            history.append(
                math.nan
            )
        else:
            history.append(
                best_evaluation
                .soft_penalty
            )

        if (
            generation
            ==
            config[
                "generations"
            ]
        ):
            break

        # --------------------------
        # Элитизм
        # --------------------------

        elite_indices = sorted(
            range(
                len(population)
            ),
            key=lambda index:
            evaluations[
                index
            ].score,
        )[
            :
            config[
                "elite_count"
            ]
        ]

        new_population = [
            population[
                index
            ][:]
            for index
            in elite_indices
        ]

        # --------------------------
        # Создание потомков
        # --------------------------

        while (
            len(
                new_population
            )
            <
            config[
                "population_size"
            ]
        ):

            parent1 = (
                tournament_selection(
                    population,
                    evaluations,
                    config,
                    rng,
                )
            )

            parent2 = (
                tournament_selection(
                    population,
                    evaluations,
                    config,
                    rng,
                )
            )

            if (
                rng.random()
                <
                config[
                    "crossover_rate"
                ]
            ):

                if (
                    crossover_kind
                    == "uniform"
                ):
                    (
                        child1,
                        child2,
                    ) = (
                        uniform_crossover(
                            parent1,
                            parent2,
                            rng,
                        )
                    )

                elif (
                    crossover_kind
                    == "onepoint"
                ):
                    (
                        child1,
                        child2,
                    ) = (
                        one_point_crossover(
                            parent1,
                            parent2,
                            rng,
                        )
                    )

                else:
                    raise ValueError(
                        "Неизвестный "
                        "crossover: "
                        f"{crossover_kind}"
                    )

            else:
                child1 = (
                    parent1[:]
                )

                child2 = (
                    parent2[:]
                )

            children = [
                mutate_schedule(
                    child1,
                    config,
                    rng,
                ),
                mutate_schedule(
                    child2,
                    config,
                    rng,
                ),
            ]

            for child in children:

                if (
                    constraint_method
                    == "repair"
                ):
                    child = (
                        repair_schedule(
                            child,
                            adjacency,
                            conflicts,
                            config,
                            rng,
                        )
                    )

                elif (
                    constraint_method
                    != "penalty"
                ):
                    raise ValueError(
                        "Неизвестный "
                        "метод ограничений: "
                        f"{constraint_method}"
                    )

                new_population.append(
                    child
                )

                if (
                    len(
                        new_population
                    )
                    >=
                    config[
                        "population_size"
                    ]
                ):
                    break

        population = (
            new_population
        )

    if (
        best_schedule is None
        or
        best_evaluation is None
    ):
        raise RuntimeError(
            "Не найдено допустимого "
            "расписания."
        )

    return GARunResult(
        best_schedule=(
            best_schedule
        ),
        best_evaluation=(
            best_evaluation
        ),
        history=history,
    )


# ============================================================
# Greedy baseline
# ============================================================

def greedy_baseline(
    adjacency,
    conflicts,
    config,
):

    exam_count = len(
        adjacency
    )

    num_slots = (
        config[
            "num_slots"
        ]
    )

    sessions_per_day = (
        config[
            "sessions_per_day"
        ]
    )

    order = sorted(
        range(
            exam_count
        ),
        key=lambda exam: (
            len(
                adjacency[
                    exam
                ]
            ),
            exam,
        ),
        reverse=True,
    )

    schedule = [
        -1
        for _ in range(
            exam_count
        )
    ]

    slot_counts = [
        0
        for _ in range(
            num_slots
        )
    ]

    for exam in order:

        variants = []

        for slot in range(
            num_slots
        ):

            hard_students = 0
            same_day_students = 0

            day = (
                slot
                // sessions_per_day
            )

            for (
                neighbour,
                shared_students,
            ) in adjacency[
                exam
            ]:

                neighbour_slot = (
                    schedule[
                        neighbour
                    ]
                )

                if neighbour_slot == -1:
                    continue

                if (
                    neighbour_slot
                    == slot
                ):
                    hard_students += (
                        shared_students
                    )

                elif (
                    neighbour_slot
                    // sessions_per_day
                    ==
                    day
                ):
                    same_day_students += (
                        shared_students
                    )

            if hard_students != 0:
                continue

            if (
                slot_counts[
                    slot
                ]
                == 0
            ):
                new_slot_penalty = (
                    config[
                        "used_slot_weight"
                    ]
                )
            else:
                new_slot_penalty = 0.0

            score = (
                config[
                    "same_day_weight"
                ]
                * same_day_students
                +
                new_slot_penalty
            )

            variants.append(
                (
                    score,
                    slot,
                )
            )

        if not variants:
            raise RuntimeError(
                "Greedy baseline "
                "не смог построить "
                "расписание."
            )

        (
            _,
            chosen_slot,
        ) = min(
            variants
        )

        schedule[
            exam
        ] = chosen_slot

        slot_counts[
            chosen_slot
        ] += 1

    evaluation = (
        evaluate_schedule(
            schedule,
            conflicts,
            config,
        )
    )

    return (
        schedule,
        evaluation,
    )


# ============================================================
# Random Feasible Search
# ============================================================

def random_feasible_search(
    adjacency,
    conflicts,
    config,
    seed,
    evaluation_budget,
):

    rng = random.Random(
        seed
    )

    best_schedule = None
    best_evaluation = None

    evaluated = 0

    while (
        evaluated
        <
        evaluation_budget
    ):

        schedule = (
            construct_feasible_schedule(
                adjacency,
                config,
                rng,
            )
        )

        if schedule is None:
            continue

        evaluation = (
            evaluate_schedule(
                schedule,
                conflicts,
                config,
            )
        )

        if not evaluation.feasible:
            continue

        evaluated += 1

        if (
            best_evaluation
            is None
            or
            evaluation.soft_penalty
            <
            best_evaluation
            .soft_penalty
        ):

            best_schedule = (
                schedule[:]
            )

            best_evaluation = (
                evaluation
            )

    return (
        best_schedule,
        best_evaluation,
    )


# ============================================================
# Статистика
# ============================================================

def calculate_statistics(
    values,
):

    return {
        "best": min(values),

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

        "worst": max(values),
    }


# ============================================================
# Подпись временного слота
# ============================================================

def slot_label(
    slot,
    config,
):

    sessions_per_day = (
        config[
            "sessions_per_day"
        ]
    )

    day = (
        slot
        // sessions_per_day
        + 1
    )

    session_index = (
        slot
        % sessions_per_day
    )

    if sessions_per_day == 2:

        if session_index == 0:
            session = "утро"
        else:
            session = "день"

    else:
        session = (
            f"сессия "
            f"{session_index + 1}"
        )

    return (
        day,
        session,
    )


# ============================================================
# Сохранение лучшего расписания
# ============================================================

def save_schedule(
    filename,
    schedule,
    evaluation,
    exams,
    student_counts,
    config,
):

    path = (
        RESULTS_DIR
        / filename
    )

    rows = sorted(
        range(
            len(schedule)
        ),
        key=lambda exam: (
            schedule[
                exam
            ],
            exams[
                exam
            ].exam_id,
        ),
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "slot",
                "day",
                "session",
                "exam_id",
                "exam_name",
                "students",
            ]
        )

        for exam_index in rows:

            slot = schedule[
                exam_index
            ]

            (
                day,
                session,
            ) = slot_label(
                slot,
                config,
            )

            exam = exams[
                exam_index
            ]

            writer.writerow(
                [
                    slot + 1,
                    day,
                    session,
                    exam.exam_id,
                    exam.name,
                    student_counts[
                        exam_index
                    ],
                ]
            )

        writer.writerow([])

        writer.writerow(
            [
                "feasible",
                int(
                    evaluation
                    .feasible
                ),
            ]
        )

        writer.writerow(
            [
                "hard_conflict_pairs",
                evaluation
                .hard_conflict_pairs,
            ]
        )

        writer.writerow(
            [
                "conflicting_students",
                evaluation
                .conflicting_students,
            ]
        )

        writer.writerow(
            [
                "same_day_students",
                evaluation
                .same_day_students,
            ]
        )

        writer.writerow(
            [
                "used_slots",
                evaluation
                .used_slots,
            ]
        )

        writer.writerow(
            [
                "soft_penalty",
                evaluation
                .soft_penalty,
            ]
        )


# ============================================================
# Примеры допустимых и недопустимых решений
# ============================================================

def save_examples(
    adjacency,
    conflicts,
    config,
):

    (
        greedy_schedule,
        _,
    ) = greedy_baseline(
        adjacency,
        conflicts,
        config,
    )

    rng = random.Random(
        config[
            "run_seed_start"
        ]
        + 100_000
    )

    second_feasible = (
        construct_feasible_schedule(
            adjacency,
            config,
            rng,
        )
    )

    if second_feasible is None:
        second_feasible = (
            greedy_schedule[:]
        )

    # Явно недопустимое:
    # все экзамены в одном слоте.
    infeasible_all_same = [
        0
        for _ in adjacency
    ]

    # Второе недопустимое:
    # берём корректное расписание
    # и искусственно создаём
    # один конфликт.
    infeasible_forced = (
        greedy_schedule[:]
    )

    (
        exam_a,
        exam_b,
        _,
    ) = conflicts[0]

    infeasible_forced[
        exam_b
    ] = (
        infeasible_forced[
            exam_a
        ]
    )

    examples = [
        (
            "feasible_greedy",
            greedy_schedule,
        ),
        (
            "feasible_randomized",
            second_feasible,
        ),
        (
            "infeasible_all_in_one_slot",
            infeasible_all_same,
        ),
        (
            "infeasible_forced_conflict",
            infeasible_forced,
        ),
    ]

    with (
        RESULTS_DIR
        / "examples.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "example",
                "feasible",
                "hard_conflict_pairs",
                "conflicting_students",
                "same_day_students",
                "used_slots",
                "soft_penalty",
                "schedule",
            ]
        )

        for (
            name,
            schedule,
        ) in examples:

            evaluation = (
                evaluate_schedule(
                    schedule,
                    conflicts,
                    config,
                )
            )

            writer.writerow(
                [
                    name,

                    int(
                        evaluation
                        .feasible
                    ),

                    evaluation
                    .hard_conflict_pairs,

                    evaluation
                    .conflicting_students,

                    evaluation
                    .same_day_students,

                    evaluation
                    .used_slots,

                    evaluation
                    .soft_penalty,

                    ";".join(
                        str(
                            slot + 1
                        )
                        for slot
                        in schedule
                    ),
                ]
            )


# ============================================================
# График сходимости
# ============================================================

def save_convergence_plot(
    histories,
    generations,
):

    plt.figure(
        figsize=(
            11,
            6,
        )
    )

    for (
        method_name,
        method_histories,
    ) in histories.items():

        means = []

        for generation in range(
            generations + 1
        ):

            values = [
                history[
                    generation
                ]
                for history
                in method_histories
                if not math.isnan(
                    history[
                        generation
                    ]
                )
            ]

            if values:
                means.append(
                    statistics.fmean(
                        values
                    )
                )
            else:
                means.append(
                    math.nan
                )

        plt.plot(
            range(
                generations + 1
            ),
            means,
            label=method_name,
        )

    plt.xlabel(
        "Поколение"
    )

    plt.ylabel(
        "Средний лучший "
        "мягкий штраф"
    )

    plt.title(
        "Сходимость ГА "
        "по 20 независимым запускам"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "convergence.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# График сравнения
# ============================================================

def save_comparison_plot(
    values_by_method,
    greedy_value,
):

    labels = list(
        values_by_method.keys()
    )

    data = [
        values_by_method[
            label
        ]
        for label
        in labels
    ]

    plt.figure(
        figsize=(
            12,
            6,
        )
    )

    plt.boxplot(
        data,
        tick_labels=labels,
    )

    plt.axhline(
        greedy_value,
        linestyle="--",
        label=(
            "Greedy = "
            f"{greedy_value:.3f}"
        ),
    )

    plt.ylabel(
        "Мягкий штраф "
        "(меньше — лучше)"
    )

    plt.title(
        "Сравнение "
        "алгоритмических вариантов"
    )

    plt.xticks(
        rotation=12
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "comparison.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# Основные эксперименты
# ============================================================

def run_experiments(
    config,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        exams,
        conflicts,
        student_counts,
    ) = load_data()

    if (
        len(exams)
        !=
        config[
            "num_exams"
        ]
    ):
        raise ValueError(
            "Количество экзаменов "
            "не соответствует config.json. "
            "Запусти: "
            "python main.py --regenerate-data"
        )

    adjacency = (
        build_adjacency(
            len(exams),
            conflicts,
        )
    )

    # --------------------------------------------------------
    # Сравнения
    # --------------------------------------------------------
    #
    # 1. repair_uniform vs penalty_uniform:
    #    меняется только обработка ограничений.
    #
    # 2. repair_uniform vs repair_onepoint:
    #    меняется только crossover.
    #

    experiment_configs = [
        (
            "GA_repair_uniform",
            "repair",
            "uniform",
        ),
        (
            "GA_penalty_uniform",
            "penalty",
            "uniform",
        ),
        (
            "GA_repair_onepoint",
            "repair",
            "onepoint",
        ),
    ]

    runs = config[
        "independent_runs"
    ]

    base_seed = config[
        "run_seed_start"
    ]

    histories = {
        name: []
        for (
            name,
            _,
            _,
        )
        in experiment_configs
    }

    values_by_method = {}

    best_by_method = {}

    run_rows = []

    # ========================================================
    # Генетический алгоритм
    # ========================================================

    for (
        method_name,
        constraint_method,
        crossover_kind,
    ) in experiment_configs:

        print()

        print(
            "=" * 68
        )

        print(
            f"Запуск серии: "
            f"{method_name}"
        )

        print(
            "=" * 68
        )

        values = []

        for run_index in range(
            runs
        ):

            seed = (
                base_seed
                + run_index
            )

            result = run_ga(
                adjacency,
                conflicts,
                config,
                seed,
                constraint_method,
                crossover_kind,
            )

            evaluation = (
                result
                .best_evaluation
            )

            values.append(
                evaluation
                .soft_penalty
            )

            histories[
                method_name
            ].append(
                result.history
            )

            run_rows.append(
                [
                    method_name,
                    run_index + 1,
                    seed,

                    int(
                        evaluation
                        .feasible
                    ),

                    evaluation
                    .hard_conflict_pairs,

                    evaluation
                    .conflicting_students,

                    evaluation
                    .same_day_students,

                    evaluation
                    .used_slots,

                    evaluation
                    .soft_penalty,
                ]
            )

            current_best = (
                best_by_method.get(
                    method_name
                )
            )

            if (
                current_best
                is None
                or
                evaluation
                .soft_penalty
                <
                current_best[
                    1
                ].soft_penalty
            ):

                best_by_method[
                    method_name
                ] = (
                    result
                    .best_schedule[:],

                    evaluation,
                )

            print(
                f"Run "
                f"{run_index + 1:02d} | "
                f"seed={seed} | "
                f"soft="
                f"{evaluation.soft_penalty:.3f} | "
                f"same_day="
                f"{evaluation.same_day_students} | "
                f"slots="
                f"{evaluation.used_slots} | "
                f"feasible="
                f"{evaluation.feasible}"
            )

        values_by_method[
            method_name
        ] = values

    # ========================================================
    # Greedy
    # ========================================================

    (
        greedy_schedule,
        greedy_eval,
    ) = greedy_baseline(
        adjacency,
        conflicts,
        config,
    )

    run_rows.append(
        [
            "Greedy",
            1,
            config[
                "data_seed"
            ],

            int(
                greedy_eval
                .feasible
            ),

            greedy_eval
            .hard_conflict_pairs,

            greedy_eval
            .conflicting_students,

            greedy_eval
            .same_day_students,

            greedy_eval
            .used_slots,

            greedy_eval
            .soft_penalty,
        ]
    )

    # ========================================================
    # Random Feasible Search
    # ========================================================

    evaluation_budget = (
        config[
            "population_size"
        ]
        *
        (
            config[
                "generations"
            ]
            + 1
        )
    )

    print()

    print(
        "=" * 68
    )

    print(
        "Запуск серии: "
        "RandomFeasibleSearch"
    )

    print(
        "=" * 68
    )

    random_values = []

    best_random = None

    for run_index in range(
        runs
    ):

        seed = (
            base_seed
            + run_index
        )

        (
            schedule,
            evaluation,
        ) = random_feasible_search(
            adjacency,
            conflicts,
            config,
            seed,
            evaluation_budget,
        )

        random_values.append(
            evaluation
            .soft_penalty
        )

        run_rows.append(
            [
                "RandomFeasibleSearch",
                run_index + 1,
                seed,

                int(
                    evaluation
                    .feasible
                ),

                evaluation
                .hard_conflict_pairs,

                evaluation
                .conflicting_students,

                evaluation
                .same_day_students,

                evaluation
                .used_slots,

                evaluation
                .soft_penalty,
            ]
        )

        if (
            best_random is None
            or
            evaluation
            .soft_penalty
            <
            best_random[
                1
            ].soft_penalty
        ):
            best_random = (
                schedule[:],
                evaluation,
            )

        print(
            f"Run "
            f"{run_index + 1:02d} | "
            f"seed={seed} | "
            f"soft="
            f"{evaluation.soft_penalty:.3f} | "
            f"same_day="
            f"{evaluation.same_day_students} | "
            f"slots="
            f"{evaluation.used_slots}"
        )

    values_by_method[
        "RandomFeasibleSearch"
    ] = random_values

    # ========================================================
    # runs.csv
    # ========================================================

    with (
        RESULTS_DIR
        / "runs.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "method",
                "run",
                "seed",
                "feasible",
                "hard_conflict_pairs",
                "conflicting_students",
                "same_day_students",
                "used_slots",
                "soft_penalty",
            ]
        )

        writer.writerows(
            run_rows
        )

    # ========================================================
    # Лучшие расписания
    # ========================================================

    for (
        method_name,
        (
            schedule,
            evaluation,
        ),
    ) in best_by_method.items():

        save_schedule(
            (
                f"best_"
                f"{method_name}.csv"
            ),
            schedule,
            evaluation,
            exams,
            student_counts,
            config,
        )

    save_schedule(
        "best_Greedy.csv",
        greedy_schedule,
        greedy_eval,
        exams,
        student_counts,
        config,
    )

    if best_random is not None:

        save_schedule(
            "best_RandomFeasibleSearch.csv",
            best_random[0],
            best_random[1],
            exams,
            student_counts,
            config,
        )

    # ========================================================
    # Два допустимых и два недопустимых примера
    # ========================================================

    save_examples(
        adjacency,
        conflicts,
        config,
    )

    # ========================================================
    # summary.txt
    # ========================================================

    summary_lines = [
        "Лабораторная работа №2",

        (
            "Вариант 19 — "
            "расписание экзаменов "
            "с конфликтами студентов"
        ),

        "",

        (
            f"Экзаменов: "
            f"{len(exams)}"
        ),

        (
            "Конфликтующих пар "
            f"экзаменов: "
            f"{len(conflicts)}"
        ),

        (
            "Временных слотов: "
            f"{config['num_slots']}"
        ),

        (
            "Сессий в день: "
            f"{config['sessions_per_day']}"
        ),

        (
            "Независимых запусков: "
            f"{runs}"
        ),

        (
            "Бюджет оценок "
            "одного запуска: "
            f"{evaluation_budget}"
        ),

        "",

        (
            "Критерий: минимизация "
            "мягкого штрафа среди "
            "допустимых расписаний."
        ),

        (
            "Жёсткое ограничение: "
            "конфликтующие экзамены "
            "не находятся "
            "в одном слоте."
        ),

        "",
    ]

    for (
        method_name,
        _,
        _,
    ) in experiment_configs:

        stats = (
            calculate_statistics(
                values_by_method[
                    method_name
                ]
            )
        )

        summary_lines.extend(
            [
                method_name,

                (
                    "  best:   "
                    f"{stats['best']:.6f}"
                ),

                (
                    "  mean:   "
                    f"{stats['mean']:.6f}"
                ),

                (
                    "  median: "
                    f"{stats['median']:.6f}"
                ),

                (
                    "  std:    "
                    f"{stats['std']:.6f}"
                ),

                (
                    "  worst:  "
                    f"{stats['worst']:.6f}"
                ),

                "",
            ]
        )

    random_stats = (
        calculate_statistics(
            random_values
        )
    )

    summary_lines.extend(
        [
            "Greedy baseline",

            (
                "  soft_penalty: "
                f"{greedy_eval.soft_penalty:.6f}"
            ),

            (
                "  same_day_students: "
                f"{greedy_eval.same_day_students}"
            ),

            (
                "  used_slots: "
                f"{greedy_eval.used_slots}"
            ),

            "",

            "RandomFeasibleSearch",

            (
                "  evaluations per run: "
                f"{evaluation_budget}"
            ),

            (
                "  best:   "
                f"{random_stats['best']:.6f}"
            ),

            (
                "  mean:   "
                f"{random_stats['mean']:.6f}"
            ),

            (
                "  median: "
                f"{random_stats['median']:.6f}"
            ),

            (
                "  std:    "
                f"{random_stats['std']:.6f}"
            ),

            (
                "  worst:  "
                f"{random_stats['worst']:.6f}"
            ),
        ]
    )

    (
        RESULTS_DIR
        / "summary.txt"
    ).write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Графики
    # ========================================================

    save_convergence_plot(
        histories,
        config[
            "generations"
        ],
    )

    save_comparison_plot(
        values_by_method,
        greedy_eval
        .soft_penalty,
    )

    # ========================================================
    # Консоль
    # ========================================================

    print()

    print(
        "=" * 68
    )

    print(
        "ИТОГОВАЯ СТАТИСТИКА"
    )

    print(
        "=" * 68
    )

    print(
        "\n".join(
            summary_lines
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
            "ЛР №2, вариант 19: "
            "генетическое построение "
            "расписания экзаменов"
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

    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help=(
            "Пересоздать "
            "синтетические "
            "входные данные"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    ensure_data(
        config,
        regenerate=(
            args.regenerate_data
        ),
    )

    run_experiments(
        config
    )


if __name__ == "__main__":
    main()