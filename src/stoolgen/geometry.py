"""Vectorized procedural geometry for a coherent family of stools.

The generator constructs point clouds directly from interpretable parts instead of
learning an implicit field. This makes ground-truth factors available for analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StoolParameters:
    seat_width: float
    seat_depth: float
    seat_thickness: float
    height: float
    leg_radius: float
    leg_spread: float
    round_seat: float
    stretcher: float

    def array(self) -> np.ndarray:
        return np.asarray(list(asdict(self).values()), dtype=np.float32)


PARAMETER_NAMES = tuple(StoolParameters.__annotations__)


def sample_parameters(rng: np.random.Generator) -> StoolParameters:
    """Draw valid geometry; correlated bounds prevent implausible proportions."""
    width = rng.uniform(0.70, 1.05)
    return StoolParameters(
        seat_width=width,
        seat_depth=rng.uniform(0.65, min(1.0, width + 0.1)),
        seat_thickness=rng.uniform(0.10, 0.20),
        height=rng.uniform(0.75, 1.25),
        leg_radius=rng.uniform(0.035, 0.075),
        leg_spread=rng.uniform(0.68, 0.90),
        round_seat=float(rng.random() < 0.35),
        stretcher=float(rng.random() < 0.55),
    )


def _box_surface(rng: np.random.Generator, count: int, size: np.ndarray, center: np.ndarray) -> np.ndarray:
    points = rng.uniform(-0.5, 0.5, (count, 3)) * size
    faces = rng.integers(0, 6, count)
    axes, signs = faces // 2, np.where(faces % 2 == 0, -1.0, 1.0)
    points[np.arange(count), axes] = signs * size[axes] / 2
    return points + center


def _cylinder_surface(
    rng: np.random.Generator, count: int, radius: float, length: float,
    center: np.ndarray, axis: int = 2,
) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, count)
    along = rng.uniform(-length / 2, length / 2, count)
    points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), along))
    if axis != 2:
        points[:, [axis, 2]] = points[:, [2, axis]]
    return points + center


def generate_stool(
    params: StoolParameters, num_points: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized points and integer part labels (seat, legs, stretchers)."""
    seat_n = int(num_points * 0.35)
    leg_n = int(num_points * 0.55)
    support_n = num_points - seat_n - leg_n
    seat_z = params.height + params.seat_thickness / 2

    if params.round_seat:
        seat = _cylinder_surface(rng, seat_n, params.seat_width / 2, params.seat_thickness,
                                 np.array([0.0, 0.0, seat_z]))
        seat[:, 1] *= params.seat_depth / params.seat_width
    else:
        seat = _box_surface(rng, seat_n,
                            np.array([params.seat_width, params.seat_depth, params.seat_thickness]),
                            np.array([0.0, 0.0, seat_z]))

    corner_x = params.seat_width * params.leg_spread / 2
    corner_y = params.seat_depth * params.leg_spread / 2
    legs = []
    per_leg = leg_n // 4
    for x, y in ((-corner_x, -corner_y), (-corner_x, corner_y),
                 (corner_x, -corner_y), (corner_x, corner_y)):
        legs.append(_cylinder_surface(rng, per_leg, params.leg_radius, params.height,
                                      np.array([x, y, params.height / 2])))
    legs_arr = np.concatenate(legs)

    supports = np.empty((0, 3), dtype=np.float64)
    if params.stretcher and support_n:
        z = params.height * 0.40
        each = support_n // 2
        a = _cylinder_surface(rng, each, params.leg_radius * 0.65, corner_x * 2,
                              np.array([0.0, -corner_y, z]), axis=0)
        b = _cylinder_surface(rng, support_n - each, params.leg_radius * 0.65, corner_y * 2,
                              np.array([corner_x, 0.0, z]), axis=1)
        supports = np.concatenate((a, b))
    else:
        # Reallocate unused support samples to the seat so every cloud has equal size.
        extra = _box_surface(rng, support_n, np.array([params.seat_width, params.seat_depth,
                                                       params.seat_thickness]),
                             np.array([0.0, 0.0, seat_z]))
        seat = np.concatenate((seat, extra))

    points = np.concatenate((seat, legs_arr, supports)).astype(np.float32)
    labels = np.concatenate((np.zeros(len(seat)), np.ones(len(legs_arr)),
                             np.full(len(supports), 2))).astype(np.int64)
    # Integer part allocations can leave up to three samples unused.
    if len(points) < num_points:
        missing = num_points - len(points)
        filler = _box_surface(rng, missing,
                              np.array([params.seat_width, params.seat_depth, params.seat_thickness]),
                              np.array([0.0, 0.0, seat_z])).astype(np.float32)
        points = np.concatenate((points, filler))
        labels = np.concatenate((labels, np.zeros(missing, dtype=np.int64)))
    # Center horizontally and map the tallest dimension to [-1, 1].
    points[:, 2] -= (params.height + params.seat_thickness) / 2
    points *= 2 / (params.height + params.seat_thickness)
    order = rng.permutation(len(points))
    return points[order][:num_points], labels[order][:num_points]
