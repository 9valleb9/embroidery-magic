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
