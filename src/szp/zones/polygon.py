"""Plan-view polygon geometry.

Written out rather than pulled from Shapely for two reasons.  The first is that
the whole geometric vocabulary needed here is four functions and a dependency
on GEOS for four functions is not worth the deployment weight on a Jetson.  The
second is that the margin semantics have to be exact and it is easier to be
sure of that with the code in front of you: dilating polygon ``A`` by ``r`` and
asking whether it meets ``B`` is the same question as asking whether the
distance from ``A`` to ``B`` is at most ``r``, which is why nothing here ever
constructs an offset polygon.  Minkowski offsets of concave polygons are fiddly
and would be a place for a bug to hide.

Everything works on concave polygons.  Zones are allowed to be L-shaped and the
unranged-detection wedges are quadrilaterals that can be very long and thin, so
convexity assumptions are not available.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "as_polygon",
    "signed_area",
    "ensure_ccw",
    "is_simple",
    "point_in_polygon",
    "points_in_polygon",
    "segments_intersect",
    "polygons_intersect",
    "polygon_distance",
    "signed_gap",
    "polygon_centroid",
    "bounding_box",
]


def as_polygon(points) -> np.ndarray:
    poly = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        raise ValueError(f"a polygon needs at least 3 vertices, got {len(poly)}")
    if np.allclose(poly[0], poly[-1]) and len(poly) > 3:
        poly = poly[:-1]
    return poly


def signed_area(poly: np.ndarray) -> float:
    """Shoelace.  Positive for counter-clockwise."""
    p = as_polygon(poly)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    p = as_polygon(poly)
    return p if signed_area(p) >= 0 else p[::-1]


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (
        c[..., 0] - a[..., 0]
    )


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float) -> np.ndarray:
    within = (
        (np.minimum(a[..., 0], b[..., 0]) - eps <= p[..., 0])
        & (p[..., 0] <= np.maximum(a[..., 0], b[..., 0]) + eps)
        & (np.minimum(a[..., 1], b[..., 1]) - eps <= p[..., 1])
        & (p[..., 1] <= np.maximum(a[..., 1], b[..., 1]) + eps)
    )
    return within


def segments_intersect(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Broadcasting segment-segment intersection test, collinear cases included."""
    d1 = _orientation(q1, q2, p1)
    d2 = _orientation(q1, q2, p2)
    d3 = _orientation(p1, p2, q1)
    d4 = _orientation(p1, p2, q2)
    s1, s2, s3, s4 = (np.sign(np.where(np.abs(d) <= eps, 0.0, d)) for d in (d1, d2, d3, d4))
    proper = (s1 * s2 < 0) & (s3 * s4 < 0)
    touching = (
        ((np.abs(d1) <= eps) & _on_segment(q1, q2, p1, eps))
        | ((np.abs(d2) <= eps) & _on_segment(q1, q2, p2, eps))
        | ((np.abs(d3) <= eps) & _on_segment(p1, p2, q1, eps))
        | ((np.abs(d4) <= eps) & _on_segment(p1, p2, q2, eps))
    )
    return proper | touching


def is_simple(poly: np.ndarray) -> bool:
    """True when no two non-adjacent edges of the polygon cross.

    Zone files are checked with this at load time.  A bow-tie polygon has an
    ambiguous interior and the occupancy answer for one would depend on which
    winding rule happened to be used.
    """
    p = as_polygon(poly)
    n = len(p)
    a, b = p, np.roll(p, -1, axis=0)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            if bool(segments_intersect(a[i], b[i], a[j], b[j])):
                return False
    return True


def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray casting for (N, 2) points.  Boundary counts as inside."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    p = as_polygon(poly)
    a = p
    b = np.roll(p, -1, axis=0)

    x = pts[:, None, 0]
    y = pts[:, None, 1]
    ax, ay = a[None, :, 0], a[None, :, 1]
    bx, by = b[None, :, 0], b[None, :, 1]

    straddles = (ay > y) != (by > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = (bx - ax) * (y - ay) / (by - ay) + ax
    crossings = straddles & (x < x_cross)
    inside = (crossings.sum(axis=1) % 2) == 1

    # A point exactly on an edge is inside.  This matters for the tests and for
    # zones that share a boundary with a wall.
    on_edge = _point_segment_distance(pts, a, b).min(axis=1) <= 1e-9
    return inside | on_edge


def point_in_polygon(point, poly: np.ndarray) -> bool:
    return bool(points_in_polygon(np.asarray(point, dtype=np.float64).reshape(1, 2), poly)[0])


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) distances from N points to M segments a[i] -> b[i]."""
    p = points[:, None, :]
    ab = (b - a)[None, :, :]
    ap = p - a[None, :, :]
    denom = np.einsum("nmk,nmk->nm", ab, ab)
    safe = np.where(denom > 0, denom, 1.0)
    t = np.where(denom > 0, np.einsum("nmk,nmk->nm", ap, ab) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=2)


def polygons_intersect(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """True when the two polygons share any area or boundary."""
    a = as_polygon(poly_a)
    b = as_polygon(poly_b)
    a1, a2 = a, np.roll(a, -1, axis=0)
    b1, b2 = b, np.roll(b, -1, axis=0)
    if bool(
        segments_intersect(
            a1[:, None, :], a2[:, None, :], b1[None, :, :], b2[None, :, :]
        ).any()
    ):
        return True
    # No crossings, so either one is wholly inside the other or they are apart.
    return bool(points_in_polygon(a[:1], b)[0] or points_in_polygon(b[:1], a)[0])


def polygon_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Least distance between two polygons, zero when they overlap."""
    a = as_polygon(poly_a)
    b = as_polygon(poly_b)
    if polygons_intersect(a, b):
        return 0.0
    b1, b2 = b, np.roll(b, -1, axis=0)
    a1, a2 = a, np.roll(a, -1, axis=0)
    return float(
        min(
            _point_segment_distance(a, b1, b2).min(),
            _point_segment_distance(b, a1, a2).min(),
        )
    )


def signed_gap(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Distance from ``poly_a`` to ``poly_b``, negative when they overlap.

    When they overlap the magnitude is how far the deepest vertex of one has
    got past the other's boundary.  That is an approximation to penetration
    depth and it is used only for reporting -- "the person is 0.4 m inside the
    zone" reads better in a log than "the person is in the zone" -- never for
    the occupancy decision itself, which is a pure boolean from
    :func:`polygons_intersect`.
    """
    a = as_polygon(poly_a)
    b = as_polygon(poly_b)
    if not polygons_intersect(a, b):
        return polygon_distance(a, b)
    b1, b2 = b, np.roll(b, -1, axis=0)
    a1, a2 = a, np.roll(a, -1, axis=0)
    # Sample along the boundaries, not only at the vertices.  Two overlapping
    # rectangles can have every vertex of each lying exactly on an edge of the
    # other and a vertex-only probe would then report zero penetration for a
    # half-metre overlap.
    a_s, b_s = _densify(a), _densify(b)
    depths = []
    inside_a = points_in_polygon(a_s, b)
    if inside_a.any():
        depths.append(_point_segment_distance(a_s[inside_a], b1, b2).min(axis=1).max())
    inside_b = points_in_polygon(b_s, a)
    if inside_b.any():
        depths.append(_point_segment_distance(b_s[inside_b], a1, a2).min(axis=1).max())
    return -float(max(depths)) if depths else 0.0


def _densify(poly: np.ndarray, samples_per_edge: int = 8) -> np.ndarray:
    """Vertices plus evenly spaced points along each edge."""
    p = as_polygon(poly)
    nxt = np.roll(p, -1, axis=0)
    ts = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)[:, None, None]
    samples = p[None, :, :] + ts * (nxt - p)[None, :, :]
    return samples.reshape(-1, 2)


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    p = as_polygon(poly)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-12:
        return p.mean(axis=0)
    cx = float(((x + xn) * cross).sum()) / (6.0 * area)
    cy = float(((y + yn) * cross).sum()) / (6.0 * area)
    return np.array([cx, cy])


def bounding_box(poly: np.ndarray) -> Tuple[float, float, float, float]:
    p = as_polygon(poly)
    return (
        float(p[:, 0].min()),
        float(p[:, 1].min()),
        float(p[:, 0].max()),
        float(p[:, 1].max()),
    )
