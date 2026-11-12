"""Routing inside one thread (embroidery/route.py): one jump per separate piece,
travel runs hidden under shapes sewn after them, trims marked."""
from shapely.geometry import Point
from shapely.ops import unary_union

from embroidery.route import route_layer
from embroidery.vectorize import Satin, satin_params


def _satin(x0, y0, x1, y1, w=1.5):
    """A straight satin column from (x0, y0) to (x1, y1)."""
    import math
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy)
    nx, ny = -dy / n * w / 2, dx / n * w / 2
    pts = [(x0 + dx * t / 4, y0 + dy * t / 4) for t in range(5)]
    sat = Satin([(x + nx, y + ny) for x, y in pts], [(x - nx, y - ny) for x, y in pts], w)
    return sat.poly, {**satin_params(w), "_satin": sat}


def _shape_poly(g, p):
    return p["_satin"].poly if "_satin" in p else g


def test_one_jump_per_piece_and_hidden_travel():
    # an "L" (stem + foot meeting end to end) and an "i" (stem + separate dot) beside it
    shapes = [_satin(0, 0, 0, 10), _satin(0, 10, 8, 10),     # L
              _satin(12, 3, 12, 10), _satin(12, 0.5, 12, 1.6)]  # i stem, i dot
    route, jumps = route_layer(shapes, Point(6, 5))
    sewn = [(g, p) for g, p in route if "_travel" not in p]
    assert len(sewn) == 4
    assert jumps == 2                                   # L -> i stem -> dot (3 pieces)
    assert sum(p.get("trim_after") == "true" for _, p in sewn) == 2
    # every travel run lies under shapes sewn after it
    for k, (g, p) in enumerate(route):
        if "_travel" in p:
            later = unary_union([_shape_poly(q, r) for q, r in route[k + 1:] if "_travel" not in r])
            assert g.difference(later).length < 0.05


def test_touching_shapes_need_no_jump():
    # an "E"-like piece: spine and three arms, all touching
    shapes = [_satin(0, 0, 0, 10), _satin(0, 0, 6, 0), _satin(0, 5, 5, 5), _satin(0, 10, 6, 10)]
    route, jumps = route_layer(shapes, Point(20, 20))
    assert jumps == 0
    assert not any(p.get("trim_after") for _, p in route)


def test_serif_feet_become_one_satin_bar():
    """An i's stem column with a slab foot: the two foot pieces left either side of the
    column become one satin bar across it instead of two grainy fills."""
    import numpy as np
    from embroidery.raster import PX_PER_MM
    from embroidery.vectorize import split_serifs
    k = PX_PER_MM
    ink = np.zeros((14 * k, 10 * k), bool)
    ink[1 * k:12 * k, 4 * k:int(5.6 * k)] = True          # 1.6 mm stem
    ink[int(10.6 * k):12 * k, int(2.6 * k):7 * k] = True  # 1.4 mm tall foot, flaring both sides
    stem = Satin([(4.0, 1.0), (4.0, 6.0), (4.0, 12.0)], [(5.6, 1.0), (5.6, 6.0), (5.6, 12.0)], 1.6)
    rest = ink.copy()
    rest[:, 4 * k:int(5.6 * k)] = False                    # what the column leaves: the two flares
    rest, bars = split_serifs(rest, [stem], ink)
    assert len(bars) == 1
    assert rest.sum() == 0
    x0, _, x1, _ = bars[0].poly.bounds
    assert x1 - x0 > 3.5                                   # the bar runs across the whole foot
