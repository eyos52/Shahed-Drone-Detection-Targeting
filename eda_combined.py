#!/usr/bin/env python3
# EDA over the combined 5-class index produced by build_dataset.py.
# Reads /content/index.csv (one row per box). Text output only, so it is
# readable over a terminal / MCP session rather than needing a rendered tab.
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
INDEX = "/content/index.csv"
INPUT_SIZE = 640          # YOLOv7-lite / yolov7_l_coco_lite.py img_scale


def load():
    df = pd.read_csv(INDEX)
    # letterbox scale: longest side -> INPUT_SIZE, matching Resize(keep_ratio=True)
    df["scale"] = INPUT_SIZE / df[["img_w", "img_h"]].max(axis=1)
    df["w_in"] = df.box_w_px * df.scale
    df["h_in"] = df.box_h_px * df.scale
    df["sqrt_in"] = np.sqrt(df.w_in * df.h_in)
    df["ar"] = df.box_w_px / df.box_h_px
    return df


def composition(df):
    print(f"rows (boxes): {len(df):,}   unique images: {df.out_name.nunique():,}\n")
    print("=== boxes: source x class ===")
    print(pd.crosstab(df.source, df.cls_name, margins=True))
    imgs = df.drop_duplicates("out_name")
    print("\n=== images: source x split ===")
    print(pd.crosstab(imgs.source, imgs.split, margins=True))
    pc = df.drop_duplicates(["out_name", "cls_name"])
    print("\n=== images per class ===")
    print(pd.crosstab(pc.cls_name, pc.split, margins=True))
    bpi = df.groupby("out_name").size()
    print("\n=== boxes per image ===")
    print(bpi.describe().round(2).to_string())
    ncls = df.groupby("out_name").cls_name.nunique()
    print(f"\nimages with >1 class: {(ncls > 1).sum():,} of {len(ncls):,}")
    res = imgs.assign(res=imgs.img_w.astype(str) + "x" + imgs.img_h.astype(str))
    print("\n=== resolution by source ===")
    print(res.groupby("source").res.agg(lambda s: s.value_counts().index[0]).to_string())


def geometry(df):
    print(f"\n=== box sqrt(area) in px at {INPUT_SIZE} input, by class ===")
    g = df.groupby("cls_name").sqrt_in
    print(pd.DataFrame({"n": g.size(), "p10": g.quantile(.10), "median": g.median(),
                        "p90": g.quantile(.90), "max": g.max()}).round(1).to_string())

    print(f"\n=== tiny-object burden at {INPUT_SIZE} ===")
    rows = []
    for cls, sub in list(df.groupby("cls_name")) + [("ALL", df)]:
        rows.append({"class": cls, "n": len(sub),
                     "<8px": f"{100*(sub.sqrt_in < 8).mean():5.1f}%",
                     "<16px": f"{100*(sub.sqrt_in < 16).mean():5.1f}%",
                     "<32px": f"{100*(sub.sqrt_in < 32).mean():5.1f}%"})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n=== COCO size buckets at {INPUT_SIZE} (% of boxes) ===")
    bucket = pd.cut(df.w_in * df.h_in, [0, 32**2, 96**2, np.inf],
                    labels=["small <32^2", "medium", "large >96^2"])
    print(pd.crosstab(df.cls_name, bucket, normalize="index").mul(100).round(1).to_string())

    print("\n=== aspect ratio (w/h) by class ===")
    g3 = df.groupby("cls_name").ar
    print(pd.DataFrame({"p10": g3.quantile(.10), "median": g3.median(),
                        "p90": g3.quantile(.90)}).round(2).to_string())

    print("\n=== box center (normalised) by class ===")
    print(df.groupby("cls_name")[["cx", "cy"]].median().round(3).to_string())


def domain_gap(df):
    # The headline risk: the shahed class is ~98% synthetic, and the synthetic
    # boxes do not look like the real ones.
    print("\n=== shahed: synthetic vs real ===")
    sh = df[df.cls_name == "shahed"].copy()
    sh["kind"] = np.where(sh.source == "shahed_synth", "synthetic", "real")
    g = sh.groupby("kind")
    print(pd.DataFrame({
        "boxes": g.size(),
        f"median_sqrt{INPUT_SIZE}": g.sqrt_in.median().round(1),
        "p10": g.sqrt_in.quantile(.10).round(1),
        "p90": g.sqrt_in.quantile(.90).round(1),
        "median_ar": g.ar.median().round(2),
        "median_cx": g.cx.median().round(3),
        "median_cy": g.cy.median().round(3),
    }).to_string())

    # size overlap between drone and shahed decides whether the model can even
    # separate them on appearance rather than on scale
    print("\n=== drone vs shahed size overlap (sqrt area at input) ===")
    for cls in ("drone", "shahed"):
        s = df[df.cls_name == cls].sqrt_in
        print(f"  {cls:9} p25={s.quantile(.25):6.1f}  median={s.median():6.1f}  "
              f"p75={s.quantile(.75):6.1f}")


if __name__ == "__main__":
    d = load()
    composition(d)
    geometry(d)
    domain_gap(d)
