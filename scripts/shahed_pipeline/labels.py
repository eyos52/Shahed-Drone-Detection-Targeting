#!/usr/bin/env python3
"""One place that knows how to read a YOLO label line.

Ten of the 28 pooled Roboflow datasets export SEGMENTATION polygons rather
than boxes, in the same .txt layout and the same directory:

    box      cls cx cy w h                      5 fields
    polygon  cls x1 y1 x2 y2 x3 y3 ...          odd, >= 7 fields

Both are "YOLO format", and nothing in the file says which you have. Reading a
polygon's first four numbers as cx/cy/w/h silently yields a tall sliver box
pinned near the left edge -- it does not error, it just quietly mislabels the
frame and corrupts every size statistic computed from it. So every consumer
in this pipeline goes through here instead of slicing p[1:5].
"""


def parse_line(line):
    """-> (cls:int, cx, cy, w, h) normalised, or None if unparseable."""
    p = line.split()
    if len(p) < 5:
        return None
    try:
        cls = int(float(p[0]))
        v = [float(x) for x in p[1:]]
    except ValueError:
        return None

    if len(v) == 4:
        cx, cy, w, h = v
    elif len(v) >= 6 and len(v) % 2 == 0:
        xs, ys = v[0::2], v[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    else:
        return None

    if w <= 0 or h <= 0:
        return None
    return cls, cx, cy, w, h


def parse_file(path):
    """-> list of (cls, cx, cy, w, h). Missing file gives []."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        r = parse_line(line)
        if r:
            out.append(r)
    return out


def is_polygon(line):
    p = line.split()
    return len(p) >= 7 and len(p) % 2 == 1
