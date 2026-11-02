"""Build a folder in the exact layout of step 1 (images/*.jpg, annotations/instances_val2017.json,
selection_log.csv) from real photographs bundled with scikit-image, with polygon objects of known area."""
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data

out = Path(sys.argv[1])
(out / "images").mkdir(parents=True, exist_ok=True)
(out / "annotations").mkdir(exist_ok=True)

sources = [data.astronaut(), data.coffee(), data.chelsea(), data.rocket(), data.immunohistochemistry(),
           np.asarray(Image.fromarray(data.camera()).convert("RGB")), data.retina()[::2, ::2], data.colorwheel()]
rng = np.random.default_rng(7)
imgs, anns, log = [], [], []
aid = 1


def ellipse(cx, cy, rx, ry, n=24):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)]).ravel().round(2).tolist()


k = 0
for rep in range(3):                      # 24 images: 8 photos x 3 crops/flips
    for src in sources:
        k += 1
        a = src if rep == 0 else (src[:, ::-1] if rep == 1 else src[::-1])
        a = np.ascontiguousarray(a[..., :3])
        h, w = a.shape[:2]
        if min(h, w) > 480:               # COCO-like size
            s = 480 / min(h, w)
            a = np.asarray(Image.fromarray(a).resize((round(w * s), round(h * s)), Image.BICUBIC))
            h, w = a.shape[:2]
        iid = 100000 + k
        name = f"{iid:012d}.jpg"
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=90)
        (out / "images" / name).write_bytes(buf.getvalue())
        imgs.append(dict(id=iid, file_name=name, width=w, height=h, license=1 + k % 8,
                         coco_url=f"http://images.cocodataset.org/val2017/{name}"))
        A = w * h
        for share in rng.choice([0.008, 0.015, 0.03, 0.06, 0.11, 0.18, 0.25], size=4, replace=False):
            r = math.sqrt(share * A / math.pi)
            cx, cy = rng.uniform(r + 2, w - r - 2), rng.uniform(r + 2, h - r - 2)
            anns.append(dict(id=aid, image_id=iid, category_id=1, iscrowd=0, area=share * A,
                             segmentation=[ellipse(cx, cy, r * 1.2, r / 1.2)], bbox=[0, 0, 0, 0]))
            aid += 1
        # a two-part object (one annotation, two polygons) and a crowd region
        r = math.sqrt(0.02 * A / math.pi)
        anns.append(dict(id=aid, image_id=iid, category_id=2, iscrowd=0, area=0,
                         segmentation=[ellipse(r + 5, r + 5, r, r), ellipse(w - r - 5, h - r - 5, r, r)],
                         bbox=[0, 0, 0, 0]))
        aid += 1
        anns.append(dict(id=aid, image_id=iid, category_id=3, iscrowd=1, area=0,
                         segmentation={"counts": [0, 10], "size": [h, w]}, bbox=[0, 0, 0, 0]))
        aid += 1
        log.append(dict(id=iid, width=w, height=h, passed=True, reason=""))

json.dump(dict(images=imgs, annotations=anns, categories=[dict(id=i, name=f"c{i}") for i in (1, 2, 3)],
               licenses=[]), open(out / "annotations" / "instances_val2017.json", "w"))
with open(out / "selection_log.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=["id", "width", "height", "passed", "reason"])
    wr.writeheader()
    wr.writerows(log)
print(len(imgs), "images,", len(anns), "annotations ->", out)
