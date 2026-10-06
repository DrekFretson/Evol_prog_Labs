from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class TrafficScenario:
    arrivals_ns: List[int]
    arrivals_ew: List[int]

    @property
    def total_arrivals(self) -> int:
        return sum(self.arrivals_ns) + sum(self.arrivals_ew)


@dataclass(frozen=True)
class TrafficMetrics:
    average_delay: float
    average_queue: float
    max_queue: int
    stops: int
    throughput: int


def generate_scenario(
    config: dict,
    seed: int,
) -> TrafficScenario:
    rng = random.Random(seed)

    sim = config["simulation"]

    horizon = int(
        sim["horizon_seconds"]
    )

    p_ns = float(
        sim[
            "arrival_probability_ns"
        ]
    )

    p_ew = float(
        sim[
            "arrival_probability_ew"
        ]
    )

    arrivals_ns = [
        1
        if rng.random() < p_ns
        else 0
        for _ in range(horizon)
    ]

    arrivals_ew = [
        1
        if rng.random() < p_ew
        else 0
        for _ in range(horizon)
    ]

    return TrafficScenario(
        arrivals_ns=arrivals_ns,
        arrivals_ew=arrivals_ew,
    )


def simulate_signal(
    genes: Sequence[int],
    scenario: TrafficScenario,
    config: dict,
) -> TrafficMetrics:
    """
    genes = [green_ns, green_ew]

    Светофор работает циклически:

    NS green
    -> transition
    -> EW green
    -> transition

    Машина считается остановившейся,
    если при прибытии горит запрещающий
    сигнал либо перед ней уже существует
    очередь.
    """

    green_ns = int(
        genes[0]
    )

    green_ew = int(
        genes[1]
    )

    sim = config[
        "simulation"
    ]

    transition = int(
        sim[
            "transition_seconds"
        ]
    )

    service_rate = int(
        sim[
            "service_rate_per_second"
        ]
    )

    cycle = (
        green_ns
        + transition
        + green_ew
        + transition
    )

    queue_ns = 0
    queue_ew = 0

    queue_sum = 0
    wait_sum = 0

    stops = 0
    throughput = 0
    max_queue = 0

    for t, (
        arrival_ns,
        arrival_ew,
    ) in enumerate(
        zip(
            scenario.arrivals_ns,
            scenario.arrivals_ew,
        )
    ):
        phase = (
            t % cycle
        )

        ns_green = (
            phase
            < green_ns
        )

        ew_green = (
            green_ns
            + transition
            <= phase
            <
            green_ns
            + transition
            + green_ew
        )

        # Если новая машина приехала
        # на красный или уже существует
        # очередь, считаем одну остановку.

        if arrival_ns:
            if (
                not ns_green
                or queue_ns > 0
            ):
                stops += (
                    arrival_ns
                )

        if arrival_ew:
            if (
                not ew_green
                or queue_ew > 0
            ):
                stops += (
                    arrival_ew
                )

        queue_ns += (
            arrival_ns
        )

        queue_ew += (
            arrival_ew
        )

        # Проезд машин через перекрёсток.

        if ns_green:
            departed = min(
                queue_ns,
                service_rate,
            )

            queue_ns -= (
                departed
            )

            throughput += (
                departed
            )

        if ew_green:
            departed = min(
                queue_ew,
                service_rate,
            )

            queue_ew -= (
                departed
            )

            throughput += (
                departed
            )

        total_queue = (
            queue_ns
            + queue_ew
        )

        queue_sum += (
            total_queue
        )

        wait_sum += (
            total_queue
        )

        max_queue = max(
            max_queue,
            total_queue,
        )

    total_arrivals = max(
        1,
        scenario.total_arrivals,
    )

    horizon = max(
        1,
        len(
            scenario.arrivals_ns
        ),
    )

    return TrafficMetrics(
        average_delay=(
            wait_sum
            / total_arrivals
        ),
        average_queue=(
            queue_sum
            / horizon
        ),
        max_queue=(
            max_queue
        ),
        stops=stops,
        throughput=throughput,
    )


def objectives(
    genes: Sequence[int],
    scenario: TrafficScenario,
    config: dict,
) -> Tuple[
    float,
    float,
    float,
]:
    """
    Три минимизируемых критерия:

    1. средняя задержка;
    2. максимальная длина очереди;
    3. количество остановок.
    """

    metrics = simulate_signal(
        genes,
        scenario,
        config,
    )

    return (
        metrics.average_delay,
        float(
            metrics.max_queue
        ),
        float(
            metrics.stops
        ),
    )