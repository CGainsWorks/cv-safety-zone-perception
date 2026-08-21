"""Polygon geometry, the zone file format and occupancy."""

from __future__ import annotations

import numpy as np
import pytest

from szp.types import Box3D, RangeSource
from szp.zones.config import ZoneConfigError, zones_from_dict
from szp.zones.occupancy import OccupancyEvaluator, box_footprint, zone_mask
from szp.zones.polygon import (
    is_simple,
    point_in_polygon,
    polygon_centroid,
    polygon_distance,
    polygons_intersect,
    signed_area,
    signed_gap,
)

SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
L_SHAPE = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], dtype=float)


def rect(cx, cy, half=0.3):
    return np.array(
        [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half],
         [cx - half, cy + half]]
    )


# -- polygon primitives ------------------------------------------------------


def test_signed_area_and_winding():
    assert signed_area(SQUARE) == pytest.approx(4.0)
    assert signed_area(SQUARE[::-1]) == pytest.approx(-4.0)


def test_point_in_polygon_handles_concavity():
    # (2, 2) is in the bounding box of the L but not inside the L.
    assert not point_in_polygon([2.0, 2.0], L_SHAPE)
    assert point_in_polygon([2.0, 0.5], L_SHAPE)
    assert point_in_polygon([0.5, 2.5], L_SHAPE)


def test_boundary_points_count_as_inside():
    assert point_in_polygon([2.0, 1.0], SQUARE)
    assert point_in_polygon([0.0, 0.0], SQUARE)


def test_self_intersecting_polygon_is_detected():
    bow_tie = np.array([[0, 0], [2, 2], [2, 0], [0, 2]], dtype=float)
    assert is_simple(SQUARE)
    assert is_simple(L_SHAPE)
    assert not is_simple(bow_tie)


def test_distance_is_zero_when_polygons_touch_and_positive_when_apart():
    assert polygon_distance(SQUARE, SQUARE + [5.0, 0.0]) == pytest.approx(3.0)
    assert polygon_distance(SQUARE, SQUARE + [2.0, 0.0]) == pytest.approx(0.0)
    assert polygons_intersect(SQUARE, SQUARE + [1.0, 0.0])


def test_signed_gap_reports_penetration_not_zero_for_a_real_overlap():
    # Every vertex of each square lies on an edge of the other, so a
    # vertex-only probe would say zero.
    assert signed_gap(SQUARE, SQUARE + [1.0, 0.0]) == pytest.approx(-1.0, abs=0.05)


def test_centroid_of_a_square():
    assert np.allclose(polygon_centroid(SQUARE), [1.0, 1.0])


# -- configuration -----------------------------------------------------------


def minimal_zone(**overrides):
    zone = {
        "id": "z",
        "kind": "stop",
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }
    zone.update(overrides)
    return {"zones": [zone]}


def test_a_valid_file_loads():
    zone_set = zones_from_dict(minimal_zone())
    assert len(zone_set) == 1
    assert zone_set["z"].is_stop
    assert zone_set[0].bit == 0


def test_empty_file_is_rejected():
    with pytest.raises(ZoneConfigError, match="no zones"):
        zones_from_dict({"zones": []})


def test_a_file_with_no_stop_zone_is_rejected():
    with pytest.raises(ZoneConfigError, match="no enabled zone"):
        zones_from_dict(minimal_zone(kind="warn"))


def test_self_intersecting_zone_is_rejected():
    with pytest.raises(ZoneConfigError, match="crosses itself"):
        zones_from_dict(minimal_zone(polygon=[[0, 0], [1, 1], [1, 0], [0, 1]]))


def test_negative_margin_is_rejected():
    with pytest.raises(ZoneConfigError, match="negative margin"):
        zones_from_dict(minimal_zone(margin_m=-0.1))


def test_debouncing_entry_on_a_stop_zone_is_rejected():
    with pytest.raises(ZoneConfigError, match="Debouncing entry"):
        zones_from_dict(minimal_zone(enter_frames=4))


def test_duplicate_ids_are_rejected():
    doc = minimal_zone()
    doc["zones"].append(dict(doc["zones"][0]))
    with pytest.raises(ZoneConfigError, match="duplicate zone id"):
        zones_from_dict(doc)


def test_too_many_zones_for_the_can_bitmap_is_rejected():
    doc = {"zones": [
        {"id": f"z{i}", "kind": "stop", "polygon": [[i, 0], [i + 1, 0], [i + 1, 1], [i, 1]]}
        for i in range(17)
    ]}
    with pytest.raises(ZoneConfigError, match="bitmap"):
        zones_from_dict(doc)


def test_polygons_are_normalised_to_counter_clockwise():
    zone_set = zones_from_dict(minimal_zone(polygon=[[0, 0], [0, 1], [1, 1], [1, 0]]))
    assert signed_area(zone_set[0].polygon) > 0


def test_the_repository_zone_file_loads(zone_set):
    ids = [z.id for z in zone_set]
    assert "machine_envelope" in ids
    assert any(z.is_stop for z in zone_set)
    assert all(z.enter_frames == 1 for z in zone_set if z.is_stop)


# -- occupancy ---------------------------------------------------------------


def box_at(x, y, sigma=0.02, height=1.75, source=RangeSource.FUSED, footprint=None):
    return Box3D(
        center=(x, y, height / 2),
        size=(0.45, 0.6, height),
        score=0.8,
        source=source,
        sigma_xy=sigma,
        footprint=footprint,
    )


def one_zone(**overrides):
    return zones_from_dict(
        minimal_zone(polygon=[[0, 0], [4, 0], [4, 4], [0, 4]], **overrides)
    )


def test_a_box_inside_the_polygon_occupies_it():
    evaluator = OccupancyEvaluator(one_zone())
    result = evaluator.evaluate([box_at(2.0, 2.0)])[0]
    assert result.occupied
    assert result.box_indices == (0,)


def test_a_box_outside_by_more_than_the_clearance_does_not():
    evaluator = OccupancyEvaluator(one_zone(margin_m=0.5, sigma_k=0.0))
    assert not evaluator.evaluate([box_at(-2.0, 2.0)])[0].occupied


def test_the_margin_dilates_the_zone():
    inside_margin = one_zone(margin_m=0.5, sigma_k=0.0)
    no_margin = one_zone(margin_m=0.0, sigma_k=0.0)
    box = box_at(-0.5, 2.0)   # footprint spans x in [-0.725, -0.275]
    assert OccupancyEvaluator(inside_margin).evaluate([box])[0].occupied
    assert not OccupancyEvaluator(no_margin).evaluate([box])[0].occupied


def test_uncertainty_makes_a_box_occupy_more_floor_not_less():
    """The central claim of the whole design: a badly ranged person is treated
    as being closer to the zone, not written off."""
    confident = one_zone(margin_m=0.0, sigma_k=2.0)
    box_close = box_at(-0.5, 2.0, sigma=0.02)
    box_vague = box_at(-0.5, 2.0, sigma=0.5)
    assert not OccupancyEvaluator(confident).evaluate([box_close])[0].occupied
    assert OccupancyEvaluator(confident).evaluate([box_vague])[0].occupied


def test_height_band_is_an_overlap_not_a_containment():
    zone_set = zones_from_dict(
        minimal_zone(polygon=[[0, 0], [4, 0], [4, 4], [0, 4]], z_min_m=0.3, z_max_m=2.2)
    )
    crouching = box_at(2.0, 2.0, height=1.0)   # spans 0.0 to 1.0
    assert OccupancyEvaluator(zone_set).evaluate([crouching])[0].occupied


def test_a_box_entirely_below_the_band_does_not_occupy():
    zone_set = zones_from_dict(
        minimal_zone(polygon=[[0, 0], [4, 0], [4, 4], [0, 4]], z_min_m=1.5, z_max_m=2.2)
    )
    low = Box3D(center=(2.0, 2.0, 0.2), size=(0.4, 0.4, 0.4), score=0.9,
                source=RangeSource.FUSED, sigma_xy=0.02)
    assert not OccupancyEvaluator(zone_set).evaluate([low])[0].occupied


def test_entry_is_immediate_and_exit_is_debounced():
    zone_set = one_zone(enter_frames=1, exit_frames=3)
    evaluator = OccupancyEvaluator(zone_set)
    assert evaluator.evaluate([box_at(2.0, 2.0)])[0].occupied
    for expected in (True, True, False):
        assert evaluator.evaluate([])[0].occupied is expected


def test_hysteresis_survives_a_single_frame_dropout():
    evaluator = OccupancyEvaluator(one_zone(exit_frames=5))
    inside = [box_at(2.0, 2.0)]
    evaluator.evaluate(inside)
    for _ in range(4):
        assert evaluator.evaluate([])[0].occupied, "a dropped detection must not clear a zone"
    evaluator.evaluate(inside)
    assert evaluator.evaluate([])[0].occupied


def test_unranged_boxes_are_flagged_when_they_are_the_only_reason():
    wedge = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
    evaluator = OccupancyEvaluator(one_zone())
    result = evaluator.evaluate(
        [box_at(2.0, 2.0, sigma=1.0, source=RangeSource.UNRANGED, footprint=wedge)]
    )[0]
    assert result.occupied
    assert result.from_unranged

    evaluator = OccupancyEvaluator(one_zone())
    result = evaluator.evaluate(
        [
            box_at(2.0, 2.0, sigma=1.0, source=RangeSource.UNRANGED, footprint=wedge),
            box_at(2.5, 2.5),
        ]
    )[0]
    assert result.occupied
    assert not result.from_unranged


def test_disabled_zones_never_occupy():
    zone_set = zones_from_dict(
        {
            "zones": [
                {"id": "on", "kind": "stop", "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
                {
                    "id": "off",
                    "kind": "stop",
                    "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]],
                    "enabled": False,
                },
            ]
        }
    )
    results = OccupancyEvaluator(zone_set).evaluate([box_at(2.0, 2.0)])
    assert results[0].occupied
    assert not results[1].occupied


def test_zone_mask_uses_file_order():
    zone_set = zones_from_dict(
        {
            "zones": [
                {"id": "a", "kind": "stop", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                {"id": "b", "kind": "stop", "polygon": [[5, 5], [6, 5], [6, 6], [5, 6]]},
            ]
        }
    )
    results = OccupancyEvaluator(zone_set).evaluate([box_at(5.5, 5.5)])
    assert zone_mask(results, zone_set) == 0b10


def test_box_footprint_falls_back_to_a_rectangle():
    footprint = box_footprint(box_at(1.0, 2.0))
    assert footprint.shape == (4, 2)
    assert np.allclose(polygon_centroid(footprint), [1.0, 2.0])


def test_the_bounding_box_shortcut_agrees_with_the_exact_test():
    """The AABB rejection is an optimisation and must never change an answer."""
    rng = np.random.default_rng(4)
    zone_set = zones_from_dict(minimal_zone(polygon=L_SHAPE.tolist(), margin_m=0.2))
    zone = zone_set[0]
    for _ in range(300):
        x, y = rng.uniform(-2.0, 5.0, 2)
        box = box_at(x, y, sigma=float(rng.uniform(0.01, 0.4)))
        fast = OccupancyEvaluator(zone_set).evaluate([box])[0].occupied
        exact = (
            polygon_distance(box_footprint(box), zone.polygon)
            <= zone.clearance_for(box.sigma_xy)
        )
        assert fast == exact, (x, y, box.sigma_xy)
