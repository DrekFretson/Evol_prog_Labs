from __future__ import annotations

import csv
import itertools
import random
from collections import Counter
from pathlib import Path


EXAM_NAMES = [
    "Математический анализ",
    "Линейная алгебра",
    "Дискретная математика",
    "Теория вероятностей",
    "Математическая статистика",
    "Алгоритмы и структуры данных",
    "Программирование на Python",
    "Объектно-ориентированное программирование",
    "Базы данных",
    "Операционные системы",
    "Компьютерные сети",
    "Архитектура ЭВМ",
    "Теория автоматов",
    "Численные методы",
    "Оптимизация",
    "Искусственный интеллект",
    "Машинное обучение",
    "Информационная безопасность",
    "Криптография",
    "Системный анализ",
    "Моделирование систем",
    "Теория управления",
    "Цифровая обработка сигналов",
    "Физика",
    "Электротехника",
    "Микропроцессорные системы",
    "Инженерия программного обеспечения",
    "Тестирование программного обеспечения",
    "Эволюционное программирование",
    "Проектирование информационных систем",
]


def generate_dataset(
    root: Path | None = None,
    seed: int = 20261019,
    num_exams: int = 30,
    num_students: int = 120,
) -> int:
    """
    Генерирует воспроизводимый набор данных:
    - экзамены;
    - записи студентов на экзамены;
    - конфликтующие пары экзаменов.
    """

    if num_exams > len(EXAM_NAMES):
        raise ValueError(
            f"Подготовлено только {len(EXAM_NAMES)} названий экзаменов, "
            f"а запрошено {num_exams}."
        )

    if root is None:
        root = Path(__file__).resolve().parent

    data_dir = root / "data"
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(seed)

    exams = [
        (
            exam_id,
            EXAM_NAMES[exam_id - 1],
        )
        for exam_id in range(
            1,
            num_exams + 1,
        )
    ]

    # Каждый студент сдаёт 2-4 экзамена.
    # Anchor-экзамен нужен, чтобы каждый экзамен
    # гарантированно получил студентов.
    enrollments = []
    student_exam_map = {}

    for student_number in range(
        1,
        num_students + 1,
    ):
        student_id = f"S{student_number:03d}"

        exam_count = rng.choices(
            [2, 3, 4],
            weights=[
                0.45,
                0.40,
                0.15,
            ],
            k=1,
        )[0]

        anchor_exam = (
            (student_number - 1)
            % num_exams
        ) + 1

        candidates = [
            exam_id
            for exam_id in range(
                1,
                num_exams + 1,
            )
            if exam_id != anchor_exam
        ]

        chosen = [anchor_exam]

        chosen.extend(
            rng.sample(
                candidates,
                exam_count - 1,
            )
        )

        chosen = sorted(chosen)

        student_exam_map[
            student_id
        ] = chosen

        for exam_id in chosen:
            enrollments.append(
                (
                    student_id,
                    exam_id,
                )
            )

    # Если студент записан на два экзамена,
    # эти экзамены конфликтуют.
    conflict_counter = Counter()

    for chosen in student_exam_map.values():

        for exam_a, exam_b in itertools.combinations(
            chosen,
            2,
        ):
            conflict_counter[
                (
                    exam_a,
                    exam_b,
                )
            ] += 1

    conflicts = [
        (
            exam_a,
            exam_b,
            shared_students,
        )
        for (
            exam_a,
            exam_b,
        ), shared_students
        in sorted(
            conflict_counter.items()
        )
    ]

    with (
        data_dir / "exams.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "exam_id",
                "exam_name",
            ]
        )

        writer.writerows(exams)

    with (
        data_dir / "enrollments.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "student_id",
                "exam_id",
            ]
        )

        writer.writerows(
            enrollments
        )

    with (
        data_dir / "conflicts.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "exam_a",
                "exam_b",
                "shared_students",
            ]
        )

        writer.writerows(
            conflicts
        )

    print(
        "Данные сгенерированы: "
        f"экзаменов={num_exams}, "
        f"студентов={num_students}, "
        f"конфликтующих пар={len(conflicts)}, "
        f"seed={seed}"
    )

    return len(conflicts)


if __name__ == "__main__":
    generate_dataset()