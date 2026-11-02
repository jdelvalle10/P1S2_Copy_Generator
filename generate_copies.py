# Copyright (c) 2026 Jose Luis del Valle <joseluisdelvalle@gmail.com>
# Released under the MIT License (see LICENSE).
"""Step 2 of the Distance-vs-Accuracy experiment: build the copies of every original.

Reads ONLY the output folder of step 1 (COCO_Dataset-Retriever):
    IN/images/*.jpg                         original COCO bytes
    IN/annotations/instances_<split>2017.json   cached COCO annotations (object outlines, crowd flags, licences);
                                            both train and val for a train+val pool
    IN/images_manifest.csv, IN/run_metadata.json   step-1 records: every JPEG is checked against its SHA-256
    IN/selection_log.csv                    (checksummed for provenance)

Writes to OUT:
    originals/<image_id>.png                decoded 8-bit RGB reference (lossless); all distances use it
    copies/<image_id>/<cell>.png            one PNG per copy (24 benign + 15 content-altering cells)
    manifest_originals.csv                  one row per original: half, sizes, licence, checksums
    manifest_copies.csv                     one row per (original, cell): status, parameters, measured area, seed, sha256
    qa/contact_<image_id>.png               contact sheets for visual QA (protocol step 7)
    run_metadata.json                       settings, catalogue, library versions, input and script checksums
    parts/<image_id>.json                   per-original records (used to resume an interrupted run)

Protocol steps covered: 4 (partition), 5 (catalogues, fixed below), 6 (generate copies), 7 (QA sheets).

    python3 generate_copies.py --in ~/COCOSubset --out ~/DVA_copies
    python3 generate_copies.py --in ~/COCOSubset --out ~/DVA_pilot --limit 100      # pilot
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, features
from pycocotools import mask as mask_utils

SCRIPT_VERSION = "2.1.0"
DEFAULT_SEED = 20261029          # same master seed as step 1

# ---------------------------------------------------------------- frozen parameters (protocol + 2 Nov 2026 confirmations)
OBJ_SHARE_MIN, OBJ_SHARE_MAX = 0.01, 0.20    # donor / copy-move objects: share of their OWN image
FEATHER_SIGMA = 0.8                          # Gaussian feather of the paste edge, px
REMOVAL_WINDOW = (0.6, 1.6)                  # applied to the mask AFTER dilation
REMOVAL_DILATE_ITER = 2                      # 3x3 kernel, twice
INPAINT_RADIUS = 3                           # cv2.inpaint radius, px (OpenCV Telea)
MAX_ATTEMPTS = 30                            # placement attempts before a cell is "not applicable"
SHARPEN = dict(radius=2, percent=100, threshold=0)
ROTATE_FILL = (0, 0, 0)                      # corners uncovered by a rotation (canvas kept)
AREAS = (0.01, 0.02, 0.05, 0.10, 0.20)


def _benign():
    c = []
    for q in (90, 70, 50):
        c.append(("jpeg_q%d" % q, "compression", "jpeg", q))
    for q in (80, 50):
        c.append(("webp_q%d" % q, "compression", "webp", q))
    for f in (0.5, 0.75, 1.5):
        c.append(("resize_x%s" % f, "scaling", "resize", f))
    for f in (1.15, 0.85):
        c.append(("brightness_%+d" % round((f - 1) * 100), "photometric", "brightness", f))
    for f in (1.15, 0.85):
        c.append(("contrast_%+d" % round((f - 1) * 100), "photometric", "contrast", f))
    for g in (0.8, 1.2):
        c.append(("gamma_%s" % g, "photometric", "gamma", g))
    for s in (2, 5):
        c.append(("noise_s%d" % s, "degradation", "noise", s))
    for s in (0.5, 1.0):
        c.append(("blur_s%s" % s, "degradation", "blur", s))
    c.append(("sharpen_100", "degradation", "sharpen", 100))
    for d in (1, 2, 5):
        c.append(("rotate_%d" % d, "mild_geometric", "rotate", d))
    for p in (0.02, 0.05):
        c.append(("crop_%d" % round(p * 100), "mild_geometric", "crop", p))
    return [dict(cell=n, stratum="benign", family=f, op=o, level=l) for n, f, o, l in c]


def _altering():
    c = []
    for op in ("splice", "removal", "copymove"):
        for a in AREAS:
            c.append(dict(cell="%s_a%d" % (op, round(a * 100)), stratum="altering", family=op, op=op, level=a))
    return c


CATALOGUE = _benign() + _altering()
assert sum(c["stratum"] == "benign" for c in CATALOGUE) == 24
assert sum(c["stratum"] == "altering" for c in CATALOGUE) == 15

COPY_FIELDS = ["image_id", "half", "cell", "stratum", "family", "op", "level", "status", "reason", "file",
               "sha256", "width", "height", "target_share", "mask_share", "changed_share", "source_ann_id",
               "donor_image_id", "donor_ann_id", "source_share", "scale", "x", "y", "attempts",
               "encoded_bytes", "seed"]
ORIG_FIELDS = ["image_id", "order", "half", "file_name", "width", "height", "license_id", "coco_url",
               "jpg_sha256", "png_sha256", "n_objects", "n_eligible_objects"]


# ---------------------------------------------------------------- helpers
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def derive_seed(master, *parts):
    """64-bit seed from SHA-256 of the master seed and the parts: independent of run order and worker count."""
    text = "|".join(str(p) for p in (master,) + parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def write_atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def ann_mask(ann, h, w):
    """Filled outline of one COCO polygon annotation (same call chain as pycocotools COCO.annToMask)."""
    rle = mask_utils.merge(mask_utils.frPyObjects(ann["segmentation"], h, w))
    return mask_utils.decode(rle).astype(bool)


def usable(ann):
    return not ann.get("iscrowd", 0) and isinstance(ann.get("segmentation"), list)


# ---------------------------------------------------------------- benign operations
def apply_benign(arr, op, level, rng):
    """Return (copy, extra) where extra holds op-specific records."""
    im = Image.fromarray(arr)
    if op in ("jpeg", "webp"):
        buf = io.BytesIO()
        im.save(buf, format="JPEG" if op == "jpeg" else "WEBP", quality=int(level))
        data = buf.getvalue()
        out = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        return out, {"encoded_bytes": len(data)}
    if op == "resize":
        w, h = im.size
        return np.asarray(im.resize((round(w * level), round(h * level)), Image.BICUBIC)), {}
    if op == "brightness":
        return np.asarray(ImageEnhance.Brightness(im).enhance(level)), {}
    if op == "contrast":
        return np.asarray(ImageEnhance.Contrast(im).enhance(level)), {}
    if op == "gamma":
        lut = np.clip(np.rint(255.0 * (np.arange(256) / 255.0) ** level), 0, 255).astype(np.uint8)
        return lut[arr], {}
    if op == "noise":
        noisy = arr.astype(np.float64) + rng.normal(0.0, level, arr.shape)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8), {}
    if op == "blur":
        return cv2.GaussianBlur(arr, (0, 0), sigmaX=level, sigmaY=level, borderType=cv2.BORDER_REFLECT_101), {}
    if op == "sharpen":
        return np.asarray(im.filter(ImageFilter.UnsharpMask(**SHARPEN))), {}
    if op == "rotate":
        return np.asarray(im.rotate(level, resample=Image.BICUBIC, expand=False, fillcolor=ROTATE_FILL)), {}
    if op == "crop":
        w, h = im.size
        k = math.sqrt(1.0 - level)
        cw, ch = round(w * k), round(h * k)
        left, top = (w - cw) // 2, (h - ch) // 2
        return np.asarray(im.crop((left, top, left + cw, top + ch))), {}
    raise ValueError(op)


# ---------------------------------------------------------------- content-altering operations
def scaled_patch(src_img, src_mask, n_target):
    """Cut the object along its outline and scale it so its mask covers about n_target pixels.

    Returns (patch_rgb float32, mask bool, alpha float32, scale), all padded by the feather margin."""
    ys, xs = np.nonzero(src_mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, mcrop = src_img[y0:y1, x0:x1], src_mask[y0:y1, x0:x1]
    scale = math.sqrt(n_target / mcrop.sum())
    nw, nh = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
    patch = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.BICUBIC)).astype(np.float32)
    m = cv2.resize(mcrop.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR) >= 0.5
    pad = int(math.ceil(3 * FEATHER_SIGMA)) + 1
    patch = np.pad(patch, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    m = np.pad(m, pad)
    alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), FEATHER_SIGMA, borderType=cv2.BORDER_CONSTANT)
    return patch, m, alpha, scale


def paste(dst, patch, alpha, x, y):
    out = dst.astype(np.float32).copy()
    ph, pw = alpha.shape
    region = out[y:y + ph, x:x + pw]
    a = alpha[..., None]
    out[y:y + ph, x:x + pw] = a * patch + (1.0 - a) * region
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def try_paste(dst, src_img, src_mask, target, rng, avoid=None):
    """One placement attempt. Returns a result dict or None if the object does not fit / overlaps `avoid`."""
    H, W = dst.shape[:2]
    patch, m, alpha, scale = scaled_patch(src_img, src_mask, target * H * W)
    ph, pw = m.shape
    if ph > H or pw > W:
        return None
    y, x = int(rng.integers(0, H - ph + 1)), int(rng.integers(0, W - pw + 1))
    placed = np.zeros((H, W), bool)
    placed[y:y + ph, x:x + pw] = m
    if avoid is not None and (placed & avoid).any():
        return None
    copy = paste(dst, patch, alpha, x, y)
    return dict(copy=copy, mask_share=placed.sum() / (H * W), scale=scale, x=x, y=y)


def do_removal(arr, objects, target):
    """Erase the real object whose DILATED mask is closest to the target share, inside the 0.6-1.6 window."""
    H, W = arr.shape[:2]
    kernel = np.ones((3, 3), np.uint8)
    best = None
    for ann_id, mask in objects:
        dil = cv2.dilate(mask.astype(np.uint8), kernel, iterations=REMOVAL_DILATE_ITER)
        share = dil.sum() / (H * W)
        if REMOVAL_WINDOW[0] * target <= share <= REMOVAL_WINDOW[1] * target:
            key = (abs(share - target), ann_id)
            if best is None or key < best[0]:
                best = (key, ann_id, dil, share)
    if best is None:
        return None
    _, ann_id, dil, share = best
    copy = cv2.inpaint(arr, dil, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return dict(copy=copy, mask_share=share, source_ann_id=ann_id)


# ---------------------------------------------------------------- worker side
_W = {}


def _init_worker(state):
    cv2.setNumThreads(1)
    _W.update(state)


def _load_ref(image_id):
    return np.asarray(Image.open(_W["out"] / "originals" / f"{image_id}.png").convert("RGB"))


def _objects(image_id, eligible_only):
    info = _W["images"][image_id]
    out = []
    for a in info["anns"]:
        m = ann_mask(a, info["height"], info["width"])
        share = m.sum() / (info["height"] * info["width"])
        if eligible_only and not (OBJ_SHARE_MIN <= share <= OBJ_SHARE_MAX):
            continue
        out.append((a["id"], m, share))
    return out


def prepare_original(image_id):
    """Phase A: decode the JPEG once, store the lossless reference, count objects."""
    info = _W["images"][image_id]
    jpg = (_W["in"] / "images" / info["file_name"]).read_bytes()
    if info["expected_sha256"] and sha256_bytes(jpg) != info["expected_sha256"]:
        raise RuntimeError(f"{info['file_name']}: SHA-256 differs from step 1's images_manifest.csv")
    pil = Image.open(io.BytesIO(jpg))
    arr = np.asarray(pil.convert("RGB"))
    if arr.shape[:2] != (info["height"], info["width"]):
        raise RuntimeError(f"{image_id}: decoded size {arr.shape[1]}x{arr.shape[0]} differs from the annotations")
    data = png_bytes(arr)
    write_atomic(_W["out"] / "originals" / f"{image_id}.png", data)
    objs = _objects(image_id, eligible_only=False)
    elig = [(aid, float(s)) for aid, _, s in objs if OBJ_SHARE_MIN <= s <= OBJ_SHARE_MAX]
    return dict(image_id=image_id, jpg_sha256=sha256_bytes(jpg), png_sha256=sha256_bytes(data),
                n_objects=len(objs), eligible=elig)


def make_copies(image_id):
    """Phase B: every catalogue cell for one original. Returns the copy records."""
    master, half = _W["seed"], _W["half"][image_id]
    arr = _load_ref(image_id)
    H, W = arr.shape[:2]
    cdir = _W["out"] / "copies" / str(image_id)
    cdir.mkdir(parents=True, exist_ok=True)
    own = None
    rows = []
    for c in CATALOGUE:
        seed = derive_seed(master, image_id, c["cell"])
        rng = np.random.default_rng(seed)
        row = {k: "" for k in COPY_FIELDS}
        row.update(image_id=image_id, half=half, cell=c["cell"], stratum=c["stratum"], family=c["family"],
                   op=c["op"], level=c["level"], seed=seed)
        res, extra = None, {}
        if c["stratum"] == "benign":
            res, extra = apply_benign(arr, c["op"], c["level"], rng)
            res = dict(copy=res)
        else:
            target = c["level"]
            row["target_share"] = target
            if own is None:
                own = _objects(image_id, eligible_only=False)
            if c["op"] == "removal":
                res = do_removal(arr, [(aid, m) for aid, m, _ in own], target)
                row["attempts"] = 1
                if res is None:
                    row.update(status="not_applicable", reason="no object whose dilated mask is within "
                               f"{REMOVAL_WINDOW[0]}-{REMOVAL_WINDOW[1]} x target")
            else:
                if c["op"] == "splice":
                    pool = _W["donors"][half]
                    pool = [d for d in pool if d[0] != image_id]
                else:
                    pool = [(image_id, aid, s) for aid, _, s in own if OBJ_SHARE_MIN <= s <= OBJ_SHARE_MAX]
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    if not pool:
                        break
                    did, aid, s = pool[int(rng.integers(0, len(pool)))]
                    if did == image_id:
                        src_img, src_mask = arr, next(m for a, m, _ in own if a == aid)
                        avoid = src_mask
                    else:
                        src_img = _load_ref(did)
                        dinfo = _W["images"][did]
                        src_mask = ann_mask(next(a for a in dinfo["anns"] if a["id"] == aid),
                                            dinfo["height"], dinfo["width"])
                        avoid = None
                    res = try_paste(arr, src_img, src_mask, target, rng, avoid)
                    row["attempts"] = attempt
                    if res is not None:
                        res.update(source_share=s, source_ann_id=aid)
                        if c["op"] == "splice":
                            res.update(donor_image_id=did, donor_ann_id=aid, source_ann_id="")
                        break
                if res is None:
                    row.update(status="not_applicable",
                               reason="no eligible object" if not pool else f"no fit in {MAX_ATTEMPTS} attempts")
        if res is not None:
            copy = res.pop("copy")
            data = png_bytes(copy)
            path = cdir / f"{c['cell']}.png"
            write_atomic(path, data)
            if _W["verify"] and not np.array_equal(np.asarray(Image.open(path).convert("RGB")), copy):
                raise RuntimeError(f"{path}: re-read pixels differ from the generated copy")
            if c["stratum"] == "altering":
                res["changed_share"] = float((copy != arr).any(axis=2).mean())
            row.update(status="ok", file=str(path.relative_to(_W["out"])), sha256=sha256_bytes(data),
                       width=copy.shape[1], height=copy.shape[0], **extra, **res)
        rows.append(row)
    part = _W["out"] / "parts" / f"{image_id}.json"
    write_atomic(part, json.dumps(rows, default=float).encode())
    return image_id, rows


# ---------------------------------------------------------------- QA contact sheets
def contact_sheet(out, image_id, rows, tile=180):
    items = [("original", out / "originals" / f"{image_id}.png")]
    items += [(r["cell"], out / r["file"]) if r["status"] == "ok" else (r["cell"] + " (n/a)", None) for r in rows]
    cols = 8
    nrows = math.ceil(len(items) / cols)
    lab = 14
    sheet = Image.new("RGB", (cols * tile, nrows * (tile + lab)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (label, p) in enumerate(items):
        x, y = (i % cols) * tile, (i // cols) * (tile + lab)
        if p is not None:
            im = Image.open(p).convert("RGB")
            im.thumbnail((tile - 4, tile - 4))
            sheet.paste(im, (x + (tile - im.width) // 2, y + lab + (tile - im.height) // 2))
        d.text((x + 3, y + 1), label, fill="black")
    sheet.save(out / "qa" / f"contact_{image_id}.png")


# ---------------------------------------------------------------- main
def find_annotations(in_dir, file_names, pool=None):
    """Load the annotation files of the step-1 pool (from its run_metadata.json; otherwise every cached
    instances_*2017.json) and merge them. Image ids must not repeat across files."""
    ann_dir = in_dir / "annotations"
    if pool:
        paths = [ann_dir / f"instances_{p}.json" for p in pool]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"Error: step 1 used {missing} but they are not in {ann_dir}.")
    else:
        paths = sorted(ann_dir.glob("instances_*2017.json"))
    if not paths:
        raise SystemExit(f"Error: no instances_*2017.json in {ann_dir} (step 1 caches it there).")
    merged, owner = {"images": [], "annotations": []}, {}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for im in data["images"]:
            if im["id"] in owner:
                raise SystemExit(f"Error: image id {im['id']} is in both {owner[im['id']]} and {p.name}.")
            owner[im["id"]] = p.name
        merged["images"] += data["images"]
        merged["annotations"] += data["annotations"]
    names = {im["file_name"] for im in merged["images"]}
    if not file_names <= names:
        raise SystemExit("Error: the annotation files in IN/annotations do not cover every image in IN/images.")
    return paths, merged


def read_step1_records(in_dir):
    """Step-1 run_metadata.json (pool) and images_manifest.csv (expected SHA-256 per file), if present."""
    meta, sums = None, {}
    if (in_dir / "run_metadata.json").exists():
        meta = json.loads((in_dir / "run_metadata.json").read_text())
    if (in_dir / "images_manifest.csv").exists():
        with open(in_dir / "images_manifest.csv") as f:
            sums = {r["file_name"]: r["bytes_sha256"] for r in csv.DictReader(f)}
    return meta, sums


def versions():
    pk = {}
    for name in ("numpy", "pillow", "opencv-contrib-python", "opencv-python", "opencv-python-headless", "pycocotools"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return dict(python=sys.version, implementation=platform.python_implementation(),
                platform=platform.platform(), machine=platform.machine(), packages=pk,
                cv2_version=cv2.__version__, libjpeg=features.version("jpg"), libwebp=features.version("webp"),
                libjpeg_turbo=features.check_feature("libjpeg_turbo"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 2: generate and store every copy of every original.")
    ap.add_argument("--in", dest="in_dir", required=True, help="step-1 output folder (has images/ and annotations/)")
    ap.add_argument("--out", dest="out_dir", required=True, help="output folder for originals, copies and manifests")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"master seed (default {DEFAULT_SEED})")
    ap.add_argument("--limit", type=int, help="process only the first N originals of the seeded order (pilot)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes (default: all cores)")
    ap.add_argument("--qa", type=int, default=50, help="number of contact sheets (protocol: 50)")
    ap.add_argument("--no-verify", action="store_true", help="skip re-reading each PNG after writing it")
    a = ap.parse_args(argv)
    t0 = time.time()
    in_dir, out = Path(a.in_dir).expanduser().resolve(), Path(a.out_dir).expanduser().resolve()

    jpgs = sorted((in_dir / "images").glob("*.jpg"))
    if not jpgs:
        raise SystemExit(f"Error: no .jpg files in {in_dir / 'images'}.")
    step1_meta, step1_sums = read_step1_records(in_dir)
    if step1_sums and set(step1_sums) != {p.name for p in jpgs}:
        raise SystemExit("Error: IN/images does not match the images listed in IN/images_manifest.csv.")
    ann_paths, coco = find_annotations(in_dir, {p.name for p in jpgs}, step1_meta and step1_meta.get("pool"))
    by_name = {im["file_name"]: im for im in coco["images"]}
    sel = {by_name[p.name]["id"] for p in jpgs}
    anns = {}
    for ann in coco["annotations"]:
        if ann["image_id"] in sel and usable(ann):
            anns.setdefault(ann["image_id"], []).append(
                {"id": ann["id"], "segmentation": ann["segmentation"], "category_id": ann["category_id"]})
    images = {}
    for p in jpgs:
        im = by_name[p.name]
        images[im["id"]] = dict(file_name=p.name, width=im["width"], height=im["height"],
                                license_id=im.get("license"), coco_url=im.get("coco_url", ""),
                                anns=sorted(anns.get(im["id"], []), key=lambda x: x["id"]),
                                expected_sha256=step1_sums.get(p.name))
    del coco

    # Partition (protocol step 4): seeded shuffle of the sorted ids, then alternate halves -> exact 50/50,
    # and any --limit prefix of the same order stays balanced and matches the full run's assignment.
    order = sorted(images)
    np.random.default_rng(derive_seed(a.seed, "partition")).shuffle(order)
    half = {iid: ("calibration" if i % 2 == 0 else "test") for i, iid in enumerate(order)}
    if a.limit:
        order = order[:a.limit]

    settings = dict(script_version=SCRIPT_VERSION, seed=a.seed, limit=a.limit, n_originals=len(order),
                    n_available=len(images), in_dir=str(in_dir), annotations=[p.name for p in ann_paths])
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_metadata.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["settings"]["seed"] != a.seed or old["settings"]["limit"] != a.limit \
                or old["settings"]["n_available"] != len(images):
            raise SystemExit("Error: OUT holds a run with different settings. Use a new output folder.")
        print("Resuming the run already in OUT.")
    for sub in ("originals", "copies", "qa", "parts"):
        (out / sub).mkdir(exist_ok=True)

    print(f"Step 2: {len(order)} originals x {len(CATALOGUE)} cells -> {out}  (workers {a.workers})")
    print("Checksumming inputs...")
    meta = dict(settings=settings, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                inputs=dict(annotations_sha256={p.name: sha256_file(p) for p in ann_paths},
                            **{f"{n.replace('.', '_')}_sha256": sha256_file(in_dir / n) if (in_dir / n).exists()
                               else None for n in ("selection_log.csv", "images_manifest.csv", "run_metadata.json")},
                            jpegs_checked_against_step1=bool(step1_sums)),
                script_sha256=sha256_file(Path(__file__)), environment=versions(),
                parameters=dict(object_share=[OBJ_SHARE_MIN, OBJ_SHARE_MAX], feather_sigma=FEATHER_SIGMA,
                                removal_window_after_dilation=list(REMOVAL_WINDOW),
                                removal_dilation="3x3 kernel, %d iterations" % REMOVAL_DILATE_ITER,
                                inpaint="OpenCV Telea, radius %d" % INPAINT_RADIUS, max_attempts=MAX_ATTEMPTS,
                                sharpen=SHARPEN, rotate_fill=ROTATE_FILL, png_compress_level=6,
                                jpeg="Pillow defaults except quality", webp="Pillow defaults except quality",
                                resize="Pillow bicubic", rotate="Pillow bicubic, canvas kept",
                                blur="OpenCV GaussianBlur, reflect-101 border",
                                gamma="out = 255*(in/255)^gamma", noise="per-channel N(0, sigma), rounded, clipped",
                                crop="centred, aspect kept, side scaled by sqrt(1 - share)",
                                partition="seeded shuffle of sorted ids, alternate halves",
                                donors="splice donors: eligible objects of other processed originals in the same half"),
                catalogue=CATALOGUE)
    write_atomic(meta_path, json.dumps(meta, indent=2).encode())

    state = {"images": {i: images[i] for i in order}, "out": out, "in": in_dir, "seed": a.seed,
             "half": half, "verify": not a.no_verify, "donors": {}}

    # Phase A: lossless references (needed by every worker, including splice donors)
    prep = {}
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(state,)) as ex:
        futs = [ex.submit(prepare_original, i) for i in order]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            prep[r["image_id"]] = r
            if k % 100 == 0 or k == len(order):
                print(f"  references {k}/{len(order)}")
    donors = {"calibration": [], "test": []}
    for iid in order:
        for aid, s in prep[iid]["eligible"]:
            donors[half[iid]].append((iid, aid, s))
    state["donors"] = donors

    # Phase B: copies
    results, todo = {}, []
    for i in order:                      # resume only originals whose record AND every listed file exist
        part = out / "parts" / f"{i}.json"
        rows = json.loads(part.read_text()) if part.exists() else None
        if rows and all(r["status"] != "ok" or (out / r["file"]).exists() for r in rows):
            results[i] = rows
        else:
            todo.append(i)
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(state,)) as ex:
        futs = [ex.submit(make_copies, i) for i in todo]
        for k, f in enumerate(as_completed(futs), 1):
            iid, rows = f.result()
            results[iid] = rows
            if k % 50 == 0 or k == len(todo):
                print(f"  copies {k}/{len(todo)}  ({time.time() - t0:.0f} s)")

    # Manifests (rows in the seeded order, cells in catalogue order)
    with open(out / "manifest_originals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ORIG_FIELDS)
        w.writeheader()
        for k, iid in enumerate(order):
            im, p = images[iid], prep[iid]
            w.writerow(dict(image_id=iid, order=k, half=half[iid], file_name=im["file_name"], width=im["width"],
                            height=im["height"], license_id=im["license_id"], coco_url=im["coco_url"],
                            jpg_sha256=p["jpg_sha256"], png_sha256=p["png_sha256"], n_objects=p["n_objects"],
                            n_eligible_objects=len(p["eligible"])))
    with open(out / "manifest_copies.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COPY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for iid in order:
            for r in results[iid]:
                w.writerow(r)

    # QA contact sheets (protocol step 7)
    qa_ids = list(order)
    np.random.default_rng(derive_seed(a.seed, "qa")).shuffle(qa_ids)
    for iid in qa_ids[:a.qa]:
        contact_sheet(out, iid, results[iid])

    # Summary
    rows = [r for iid in order for r in results[iid]]
    na = {}
    for r in rows:
        if r["status"] != "ok":
            na[r["cell"]] = na.get(r["cell"], 0) + 1
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"Done: {ok} copies written, {len(rows) - ok} not applicable, {time.time() - t0:.0f} s.")
    for cell, n in sorted(na.items(), key=lambda kv: -kv[1]):
        print(f"  not applicable {n:5d}  {cell}")
    meta["summary"] = dict(copies_ok=ok, not_applicable=na, seconds=round(time.time() - t0, 1))
    write_atomic(meta_path, json.dumps(meta, indent=2).encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
