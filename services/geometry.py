from typing import Iterable, Tuple


def point_inside_polygon(
    px: float, py: float, poly: Iterable[Tuple[float, float]]
) -> bool:
    # Ray casting; inclusive on edges
    inside = False
    pts = list(poly)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            xin = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1
            if px <= xin:
                inside = not inside
    return inside
