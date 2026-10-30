"""MapAnything -> COLMAP model, with shared camera intrinsics (runs inside the Modal job).

    python -m movable_reality.ma_colmap --images_dir frames --output_dir recon \
        [--fx .. --fy .. --cx .. --cy ..]

MapAnything's demo_colmap.py lets the model guess intrinsics per image. On blender01 (31 Oct)
that gave poses at half metric scale and 0.43 m ATE. A phone video comes from one camera, so:
- with known intrinsics (full-resolution pixels), every view gets them as input;
- without, a first pass predicts per-view intrinsics, their median (mapped back to full
  resolution) becomes the shared camera, and a second pass runs with it.
Needs the mapanything package (Modal image only). Apache weights: facebook/map-anything-apache.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

MODEL = "facebook/map-anything-apache"


def full_res_k(k_small: np.ndarray, small_wh, full_wh) -> np.ndarray:
    """Invert MapAnything's scale + floor + centre crop (same maths as recon.full_res_camera)."""
    from .coverage.cameras import Camera  # noqa: PLC0415
    from .recon import full_res_camera  # noqa: PLC0415

    cam = Camera("x", int(small_wh[0]), int(small_wh[1]), k_small, np.eye(3), np.zeros(3))
    return full_res_camera(cam, int(full_wh[0]), int(full_wh[1])).K


def main() -> None:
    import torch  # noqa: PLC0415
    from mapanything.models import MapAnything  # noqa: PLC0415
    from mapanything.utils.colmap_export import export_predictions_to_colmap  # noqa: PLC0415
    from mapanything.utils.image import load_images, preprocess_inputs  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    for k in ("fx", "fy", "cx", "cy"):
        ap.add_argument(f"--{k}", type=float, default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")))
    names = [os.path.basename(p) for p in paths]
    W, H = Image.open(paths[0]).size
    model = MapAnything.from_pretrained(MODEL).to("cuda").eval()
    kw = dict(memory_efficient_inference=True, minibatch_size=1, use_amp=True, amp_dtype="bf16",
              apply_mask=True, mask_edges=True)  # fmt: skip
    report: dict = {"frames": len(paths), "width": W, "height": H}

    if args.fx is None:
        t0 = time.time()
        views = load_images(paths)
        with torch.no_grad():
            out = model.infer(views, **kw)
        ks = []
        for o, v in zip(out, views, strict=True):
            h, w = v["img"].shape[-2:]
            ks.append(full_res_k(o["intrinsics"][0].float().cpu().numpy(), (w, h), (W, H)))
        ks = np.array(ks)
        K = np.median(ks, axis=0)
        report["calibration"] = {
            "mode": "self, median of pass 1",
            "fx_spread_pct": float(100 * ks[:, 0, 0].std() / K[0, 0]),
            "pass1_s": round(time.time() - t0, 1),
        }
        del out, views
        torch.cuda.empty_cache()
    else:
        K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]])
        report["calibration"] = {"mode": "given"}
    report["K_full_res"] = K.tolist()
    print("shared intrinsics (full res):", np.round(K, 2).tolist(), flush=True)

    t0 = time.time()
    Kt = torch.tensor(K, dtype=torch.float32)
    raw = [
        {"img": torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))), "intrinsics": Kt}
        for p in paths
    ]
    views = preprocess_inputs(raw)
    with torch.no_grad():
        out = model.infer(views, **kw)
    report["pass2_s"] = round(time.time() - t0, 1)
    os.makedirs(args.output_dir, exist_ok=True)
    export_predictions_to_colmap(
        outputs=out,
        processed_views=views,
        image_names=names,
        output_dir=args.output_dir,
        voxel_fraction=0.01,
        data_norm_type=model.encoder.data_norm_type,
        save_ply=True,
        save_images=True,
    )
    with open(os.path.join(args.output_dir, "mapanything.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
