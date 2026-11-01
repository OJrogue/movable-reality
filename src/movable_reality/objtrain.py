"""Per-object training: every object and the background trained on their own pixels only.

    python -m movable_reality.objtrain --scene_dir /data/scenes/<id> [--obj_steps 4000]

The joint splat (splat/scene.ply) is cut into objects by a per-Gaussian vote
(scripts/lift_ids.py). Its Gaussians were fitted to whole frames, so an object's splats
carry bits of the wall and floor behind it, and the background carries the object's colour
where it stood. Moving an object then leaves a ghost behind and drags background with it
(OJ, 31 Oct). Here each part starts from its lifted splats and is fine-tuned alone:

- the cut: a background splat that lands on object k in at least half the views that see
  it while SAM found k moves to k when it is within 15 cm of its splats. One that
  lands on objects in 90% of all its views joins the nearest object within 15 cm.
  SAM ids the lift left out join the scene object of the same noun.
- junk: splats no object took that lie inside an object's box (1st to 99th percentile,
  plus 2 cm) above the floor go, when they land on objects in most views or on one
  object in a quarter of the views that see it.
- holes (--holes 1, off by default): on every 6th frame the background's depth and colour
  are inpainted over the object masks and back-projected as new splats where the
  background is thin. On blender01 (31 Oct) the fill floated in the room as speckles,
  so a wall never seen behind a sofa stays thin for now.
- in front: background splats in front of the trained objects' rendered surface, in at
  least half the views where they land on an object, go. They would have hidden the
  object in the capture.
- duplicates: two trained objects whose boxes overlap (60% of the smaller one) are one
  thing under two nouns, and install() hides the smaller.
- object o: rendered alone on black, compared with frame * mask_o, plus an opacity term
  that pulls alpha to mask_o. Pixels of other objects (which may cover o, grown by 3 px),
  of any other SAM item, unlabelled pixels (an item no noun named) and a 2 px band on the
  mask edge are ignored. Floor and wall pixels are not. Splats
  outside the object fade and are pruned.
- background: rendered alone, compared with the frame outside the objects' masks (grown
  by 6 px). A second, colour-only pass adds the frame inpainted where the objects are,
  at full weight inside the masks. Object colour leaves the walls and the floor, and the
  unseen parts behind the objects take the surrounding colour.

Run it with `modal run modal_jobs/reconstruct.py --scene <id> --objects` after lift_ids.py,
which uploads the inputs, trains on an L40S and installs the result with install() below.

Masks are SAM 3.1's per-frame labels (ids/masks, value = label + 1) and
ids/object_labels.json maps each scene object to its values. Cameras are recon_full/
(full-resolution frames, the frame of the PLYs). Writes objects_trained/<same names>.ply
and objects_trained/report.json. Needs CUDA, gsplat and fused-ssim (the Modal image).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .coverage.colmap import read_model
from .splat_ops import read_vertices, standard, write_vertices

SH_REST = 45  # degree 3


def load_params(path: Path, device):
    import torch  # noqa: PLC0415

    v = standard(read_vertices(path))
    rest = np.stack([v[f"f_rest_{i}"] for i in range(SH_REST)], 1).reshape(-1, 3, 15)
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)  # noqa: E731
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(t(np.stack([v["x"], v["y"], v["z"]], 1))),
        "scales": torch.nn.Parameter(t(np.stack([v[f"scale_{i}"] for i in range(3)], 1))),
        "quats": torch.nn.Parameter(t(np.stack([v[f"rot_{i}"] for i in range(4)], 1))),
        "opacities": torch.nn.Parameter(t(v["opacity"])),
        "sh0": torch.nn.Parameter(t(np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[:, None])),
        "shN": torch.nn.Parameter(t(rest.transpose(0, 2, 1))),  # [N, 15, 3]
    })  # fmt: skip


def save_params(params, path: Path) -> int:
    """Write the standard 3DGS layout (same as splat_ops.standard), normals 0."""
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    n = len(p["means"])
    rest = p["shN"].transpose(0, 2, 1).reshape(n, SH_REST)  # back to f_rest order
    fields = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
              + [f"f_rest_{i}" for i in range(SH_REST)]
              + ["opacity", "scale_0", "scale_1", "scale_2"]
              + ["rot_0", "rot_1", "rot_2", "rot_3"])  # fmt: skip
    out = np.zeros(n, dtype=[(f, "f4") for f in fields])
    out["x"], out["y"], out["z"] = p["means"].T
    for i in range(3):
        out[f"f_dc_{i}"] = p["sh0"][:, 0, i]
        out[f"scale_{i}"] = p["scales"][:, i]
    for i in range(SH_REST):
        out[f"f_rest_{i}"] = rest[:, i]
    for i in range(4):
        out[f"rot_{i}"] = p["quats"][:, i]
    out["opacity"] = p["opacities"]
    write_vertices(path, out)
    return n


def grow(mask, px: int):
    """Binary dilation by px pixels (max pool). mask: [H, W] bool tensor."""
    import torch.nn.functional as F  # noqa: PLC0415

    if px <= 0:
        return mask
    m = mask[None, None].float()
    return F.max_pool2d(m, 2 * px + 1, stride=1, padding=px)[0, 0] > 0.5


def shrink(mask, px: int):
    return ~grow(~mask, px)


STRUCTURE = ("floor", "wall", "ceiling")


def fold_labels(labels: dict[str, list[int]], sam: dict[str, str]) -> tuple[dict, dict]:
    """Add SAM ids the lift left out to the scene object of the same noun, when there is
    exactly one. SAM gives one object a new id after it leaves the view and comes back
    (blender01: coffee table#1 is coffee table#0 again), and the lift keeps only one.
    sam: ids/masks/labels.json, {"<value>": "<noun>#<id>"}. Returns (labels, folded)."""
    out = {o: list(v) for o, v in labels.items()}
    used = {v for vs in labels.values() for v in vs}
    folded: dict[str, list[int]] = {}
    for val, name in sam.items():
        noun = name.split("#")[0]
        if int(val) in used or noun in STRUCTURE:
            continue
        same = [o for o in out if o.split("#")[0] == noun]
        if len(same) == 1:
            out[same[0]].append(int(val))
            folded.setdefault(same[0], []).append(int(val))
    return out, folded


def label_hits(means, cams, obj_index, obj_masks, n_obj: int):
    """Per Gaussian centre: views in frame, views outside every (grown) object mask, and
    per object k (column k + 1) the views where the centre lands on k and the views that
    see it while SAM found k somewhere in the frame. Only those views can say whether a
    splat belongs to k: SAM loses objects in some frames, and a centre there looks free.
    obj_index[i]: [H, W] uint8 map to object index + 1. Occlusion is not tested."""
    import torch  # noqa: PLC0415

    n = len(means)
    n_in = torch.zeros(n, dtype=torch.int32, device=means.device)
    n_free = torch.zeros_like(n_in)
    hits = torch.zeros((n, n_obj + 1), dtype=torch.int32, device=means.device)
    seen = torch.zeros_like(hits)
    for cam, oi, om in zip(cams, obj_index, obj_masks, strict=True):
        vm, K = cam["viewmat"], cam["K"]
        pc = means @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        uv = (pc[:, :2] / z.clamp(min=1e-6)[:, None]) @ K[:2, :2].T + K[:2, 2]
        u, v = uv[:, 0].long(), uv[:, 1].long()
        ok = (z > 0.05) & (u >= 0) & (u < cam["W"]) & (v >= 0) & (v < cam["H"])
        idx = ok.nonzero()[:, 0]
        n_in[idx] += 1
        n_free[idx] += (~om[v[idx], u[idx]]).int()
        hits[idx, oi[v[idx], u[idx]].long()] += 1
        present = torch.zeros(n_obj + 1, dtype=torch.int32, device=means.device)
        present[oi.unique().long()] = 1
        seen[idx] += present
    return n_in, n_free, hits, seen


def struct_share(means, cams, struct_masks):
    """Per Gaussian centre: the share of the views in frame where it lands on a floor, wall or
    ceiling pixel. Wall is segmented on every third frame only, so a wall splat scores about
    a third; an item no noun named scores near 0 (its pixels are unlabelled)."""
    import torch  # noqa: PLC0415

    n_in = torch.zeros(len(means), dtype=torch.int32, device=means.device)
    n_st = torch.zeros_like(n_in)
    for cam, sm in zip(cams, struct_masks, strict=True):
        vm, K = cam["viewmat"], cam["K"]
        pc = means @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        uv = (pc[:, :2] / z.clamp(min=1e-6)[:, None]) @ K[:2, :2].T + K[:2, 2]
        u, v = uv[:, 0].long(), uv[:, 1].long()
        ok = (z > 0.05) & (u >= 0) & (u < cam["W"]) & (v >= 0) & (v < cam["H"])
        idx = ok.nonzero()[:, 0]
        n_in[idx] += 1
        n_st[idx] += sm[v[idx], u[idx]].int()
    return n_st.float() / n_in.clamp(min=1).float()


ITEM_STRUCT_SHARE = 0.15  # below this, an in-box background splat is an unnamed item


BG_FILL_WEIGHT = 1.0  # colour-only pass, so the fill cannot add geometry


def inpaint(img, mask, scale: int = 4):
    """Telea inpainting of mask in img ([H, W, 3] uint8 and [H, W] bool, on the GPU), run at
    1/scale resolution: only the smooth fill is wanted, and full-size masks take seconds."""
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    h, w = mask.shape
    small = cv2.resize(img.cpu().numpy(), (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.cpu().numpy().astype(np.uint8), (w // scale, h // scale),
                   interpolation=cv2.INTER_NEAREST)  # fmt: skip
    filled = cv2.inpaint(small, cv2.dilate(m, np.ones((3, 3), np.uint8)), 5, cv2.INPAINT_TELEA)
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(up, device=img.device)


def box_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller of two axis-aligned boxes, [2, 3] (min row, max row)."""
    inter = np.clip(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]), 0, None).prod()
    vol = lambda x: np.clip(x[1] - x[0], 0, None).prod()  # noqa: E731
    small = min(vol(a), vol(b))
    return float(inter / small) if small > 0 else 0.0


def merge_duplicates(verts: dict[str, np.ndarray], share: float = 0.6) -> dict[str, str]:
    """Scene objects that are one thing under two nouns (blender01: the armchair is both
    sofa#1 and armchair#0). Boxes are the 5th to 95th percentile of the splat centres;
    a pair whose intersection covers `share` of the smaller box is one object, and the one
    with fewer splats is absorbed. Objects that rest on others overlap little: books on a
    cabinet share its top face only. Returns {absorbed: kept}."""
    boxes = {}
    for o, v in verts.items():
        if len(v) >= 50:
            xyz = np.stack([v["x"], v["y"], v["z"]], 1)
            boxes[o] = np.percentile(xyz, [5, 95], axis=0)
    order = sorted(boxes, key=lambda o: -len(verts[o]))
    out: dict[str, str] = {}
    for i, big in enumerate(order):
        if big in out:
            continue
        for small in order[i + 1 :]:
            if small not in out and box_overlap(boxes[big], boxes[small]) > share:
                out[small] = big
    return out


def in_front(means, objects, cams, margin: float = 0.02, alpha_min: float = 0.5):
    """True for centres that lie in front of the objects' rendered surface (depth minus
    margin, reconstruction units) in at least half the views where they land on an object
    pixel (alpha over alpha_min). objects: one parameter dict holding every object."""
    import torch  # noqa: PLC0415
    from gsplat import rasterization  # noqa: PLC0415

    n_on = torch.zeros(len(means), dtype=torch.int32, device=means.device)
    n_front = torch.zeros_like(n_on)
    colors = objects["sh0"][:, 0]  # colour is not used, only the expected depth
    for cam in cams:
        out, alpha, _ = rasterization(
            objects["means"], objects["quats"], torch.exp(objects["scales"]),
            torch.sigmoid(objects["opacities"]), colors, cam["viewmat"][None], cam["K"][None],
            cam["W"], cam["H"], render_mode="RGB+ED", packed=False,
        )  # fmt: skip
        depth, alpha = out[0, ..., 3], alpha[0, ..., 0]
        vm, K = cam["viewmat"], cam["K"]
        pc = means @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        uv = (pc[:, :2] / z.clamp(min=1e-6)[:, None]) @ K[:2, :2].T + K[:2, 2]
        u, v = uv[:, 0].long(), uv[:, 1].long()
        ok = (z > 0.05) & (u >= 0) & (u < cam["W"]) & (v >= 0) & (v < cam["H"])
        idx = ok.nonzero()[:, 0]
        on = alpha[v[idx], u[idx]] > alpha_min
        idx, d = idx[on], depth[v[idx[on]], u[idx[on]]]
        n_on[idx] += 1
        n_front[idx] += (z[idx] < d - margin).int()
    return (n_on >= 2) & (n_front * 2 >= n_on)


def hole_points(params, cams, imgs, obj_masks, every: int = 6, stride: int = 4,
                voxel: float = 0.012):  # fmt: skip
    """New splats for the wall and floor behind objects, which no view saw. In every
    `every`-th frame the background's expected depth is rendered, its depth and colour are
    inpainted (OpenCV Telea) from the pixels where it is solid and no object covers it, and
    object pixels where the background behind is thin (alpha < 0.9) are back-projected
    (every `stride`-th pixel), one splat per `voxel` cell (reconstruction units). Returns
    (means, rgb, radius) tensors."""
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from gsplat import rasterization  # noqa: PLC0415

    pts, cols, rads = [], [], []
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    for i in range(0, len(cams), every):
        cam = cams[i]
        with torch.no_grad():
            out, alpha, _ = rasterization(
                params["means"], params["quats"], torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), colors, cam["viewmat"][None],
                cam["K"][None], cam["W"], cam["H"], sh_degree=3, render_mode="RGB+ED",
                packed=False,
            )  # fmt: skip
        a = alpha[0, ..., 0]
        depth = out[0, ..., 3].cpu().numpy().astype(np.float32)  # ED is already per alpha
        unknown = (obj_masks[i] | (a < 0.5)).cpu().numpy()  # not usable as inpaint source
        thin = (obj_masks[i] & (a < 0.9)).cpu().numpy()  # object pixels with no wall behind
        if unknown.mean() > 0.6 or not thin.any():
            continue
        h, w = unknown.shape
        sw, sh = w // stride, h // stride
        d_s = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_AREA)
        k3 = np.ones((3, 3), np.uint8)
        m_s = cv2.dilate(cv2.resize(unknown.astype(np.uint8), (sw, sh),
                                    interpolation=cv2.INTER_NEAREST), k3)  # fmt: skip
        t_s = cv2.resize(thin.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
        d_s = cv2.inpaint(d_s, m_s, 5, cv2.INPAINT_TELEA)
        rgb = cv2.resize(imgs[i].cpu().numpy(), (sw, sh), interpolation=cv2.INTER_AREA)
        rgb = cv2.inpaint(rgb, m_s, 5, cv2.INPAINT_TELEA)
        solid = d_s[(m_s == 0) & (d_s > 0.05)]
        if len(solid) < 100:
            continue
        ys, xs = np.nonzero(t_s)
        z = d_s[ys, xs]
        # Walls and floor behind objects are far: a fill nearer than most of the solid
        # background ran off a near edge and would float in the room.
        ok = z > np.percentile(solid, 10)
        ys, xs, z = ys[ok], xs[ok], z[ok]
        K = cam["K"].cpu().numpy()
        u, v = (xs + 0.5) * stride, (ys + 0.5) * stride
        pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
        vm = cam["viewmat"].cpu().numpy()
        pw = (pc - vm[:3, 3]) @ vm[:3, :3]  # camera to reconstruction frame
        pts.append(pw)
        cols.append(rgb[ys, xs] / 255.0)
        rads.append(z * stride / K[0, 0])
    if not pts:
        dev = params["means"].device
        return (torch.zeros((0, 3), device=dev),) * 2 + (torch.zeros(0, device=dev),)
    pts, cols, rads = np.concatenate(pts), np.concatenate(cols), np.concatenate(rads)
    # Inpainted depth can run off at a mask edge: keep points inside the room's splats.
    ref = params["means"][:: max(1, len(params["means"]) // 200_000)].detach().cpu().numpy()
    lo, hi = np.percentile(ref, 1, axis=0), np.percentile(ref, 99, axis=0)
    inside = np.all((pts > lo) & (pts < hi), axis=1)
    pts, cols, rads = pts[inside], cols[inside], rads[inside]
    _, first = np.unique(np.floor(pts / voxel).astype(np.int64), axis=0, return_index=True)
    dev = params["means"].device
    t = lambda x: torch.tensor(x[first], dtype=torch.float32, device=dev)  # noqa: E731
    return t(pts), t(cols), t(rads)


def add_splats(params, means, rgb, radius, opacity: float = 0.9):
    """Append isotropic splats (SH degree 0) to a parameter dict."""
    import torch  # noqa: PLC0415

    n = len(means)
    new = {
        "means": means,
        "scales": torch.log(radius.clamp(min=1e-4))[:, None].repeat(1, 3),
        "quats": torch.tensor([1.0, 0, 0, 0], device=means.device).repeat(n, 1),
        "opacities": torch.full((n,), float(np.log(opacity / (1 - opacity))), device=means.device),
        "sh0": ((rgb - 0.5) / 0.28209479177387814)[:, None],
        "shN": torch.zeros((n, 15, 3), device=means.device),
    }
    for k in list(params.keys()):
        params[k] = torch.nn.Parameter(torch.cat([params[k].detach(), new[k]]))


FLOOR_CLEAR = 0.03  # world metres: splats lower than this are floor, whatever box they are in
CLAIM_DIST_M = 0.15  # world metres, converted with the scale of world_from_recon
FRONT_MARGIN_M = 0.03


def nearest_dist(points, ref, chunk: int = 4096):
    """Distance from each point to its nearest ref point (brute force in chunks, on the GPU)."""
    import torch  # noqa: PLC0415

    if len(ref) == 0:
        return torch.full((len(points),), float("inf"), device=points.device)
    out = [torch.cdist(points[i : i + chunk], ref).min(1).values
           for i in range(0, len(points), chunk)]  # fmt: skip
    return torch.cat(out) if out else torch.zeros(0, device=points.device)


def train_part(params, frames, cams, weight_fn, steps: int, scene_scale: float, device,
               alpha_target=None, alpha_lambda: float = 0.0, log: str = "",
               colour_only: bool = False) -> dict:  # fmt: skip
    """Fine-tune one part. frames: list of indices into cams (images on the GPU as uint8).
    weight_fn(i) -> (w [H, W] float, target [H, W, 3] float). alpha_target(i) -> (a, wa).
    colour_only trains the SH colours alone, with no densification and no prune."""
    import torch  # noqa: PLC0415
    from gsplat import rasterization  # noqa: PLC0415
    from gsplat.strategy import DefaultStrategy  # noqa: PLC0415

    try:
        from fused_ssim import fused_ssim  # noqa: PLC0415
    except ImportError:  # pragma: no cover - the Modal image has it
        fused_ssim = None
    lrs = {"means": 1.6e-5 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}  # fmt: skip
    if colour_only:  # base colour only: unseen splats have no views to fit SH to
        lrs = {"sh0": 4 * lrs["sh0"]}
        for k in params:  # frozen: no gradients piling up unused
            params[k].requires_grad_(k == "sh0")
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
            for k, lr in lrs.items()}  # fmt: skip
    sched = torch.optim.lr_scheduler.ExponentialLR(
        opts["sh0" if colour_only else "means"], gamma=0.1 ** (1 / steps)
    )
    strategy = DefaultStrategy(absgrad=True, grow_grad2d=0.0006, refine_start_iter=300,
                               refine_stop_iter=int(0.7 * steps), reset_every=10**9,
                               refine_every=100, verbose=False)  # fmt: skip
    if not colour_only:
        strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)
    n0 = len(params["means"])
    rng = np.random.default_rng(0)
    t0 = time.time()
    loss_sum = 0.0
    for step in range(steps):
        i = frames[rng.integers(len(frames))]
        cam = cams[i]
        w, target = weight_fn(i)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        img, alpha, info = rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors, cam["viewmat"][None], cam["K"][None],
            cam["W"], cam["H"], sh_degree=3, packed=False, absgrad=True,
            rasterize_mode="antialiased",
        )  # fmt: skip
        if not colour_only:
            strategy.step_pre_backward(params, opts, state, step, info)
        pred = img[0].clamp(0, 1)
        wm = w[..., None]
        l1 = ((pred - target).abs() * wm).sum() / (wm.sum() * 3 + 1e-6)
        loss = 0.8 * l1
        if fused_ssim is not None:  # ignored pixels take the target, so they add no error
            mixed = pred * wm + target * (1 - wm)
            ssim = fused_ssim(mixed.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None],
                              padding="valid")  # fmt: skip
            loss = loss + 0.2 * (1 - ssim)
        else:
            loss = loss + 0.2 * l1
        if alpha_target is not None:
            a, wa = alpha_target(i)
            la = ((alpha[0, ..., 0] - a).abs() * wa).sum() / (wa.sum() + 1e-6)
            loss = loss + alpha_lambda * la
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        if not colour_only:
            strategy.step_post_backward(params, opts, state, step, info, packed=False)
        loss_sum += float(loss)
        if (step + 1) % 1000 == 0:
            print(f"  {log} step {step + 1}/{steps} loss {loss_sum / 1000:.4f} "
                  f"{len(params['means'])} GSs", flush=True)  # fmt: skip
            loss_sum = 0.0
    # Final prune: splats that faded out (the alpha term drives non-object splats to 0).
    with torch.no_grad():
        keep = torch.sigmoid(params["opacities"]) > (0.0 if colour_only else 0.02)
        for k in list(params.keys()):
            params[k] = torch.nn.Parameter(params[k][keep])
    return {"gaussians_in": n0, "gaussians_out": int(keep.sum()), "frames": len(frames),
            "steps": steps, "seconds": round(time.time() - t0, 1)}  # fmt: skip


def main() -> None:
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=Path, required=True)
    ap.add_argument("--obj_steps", type=int, default=4000)
    ap.add_argument("--bg_steps", type=int, default=8000)
    ap.add_argument("--fill_steps", type=int, default=3000, help="colour-only fill pass")
    ap.add_argument(
        "--holes", type=int, default=0, help="1: add inpainted hole splats (experimental)"
    )
    ap.add_argument("--min_px", type=int, default=800, help="mask pixels for a frame to count")
    a = ap.parse_args()
    root, dev = a.scene_dir, "cuda"
    scene = json.loads((root / "scene.json").read_text())
    labels = json.loads((root / "ids" / "object_labels.json").read_text())
    out = root / "objects_trained"
    out.mkdir(exist_ok=True)

    cams, imgs, masks = [], [], []
    for c in read_model(root / "recon_full"):
        stem = Path(c.name).stem
        mpath = root / "ids" / "masks" / f"{stem}.png"
        if not mpath.exists():
            continue
        vm = np.eye(4)
        vm[:3, :3], vm[:3, 3] = c.R, c.t
        cams.append({"viewmat": torch.tensor(vm, dtype=torch.float32, device=dev),
                     "K": torch.tensor(c.K, dtype=torch.float32, device=dev),
                     "W": c.width, "H": c.height, "centre": c.center})  # fmt: skip
        imgs.append(torch.tensor(np.asarray(Image.open(root / "frames" / c.name).convert("RGB")),
                                 device=dev))  # fmt: skip
        masks.append(torch.tensor(np.asarray(Image.open(mpath)), device=dev))
    centres = np.array([c["centre"] for c in cams])
    scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    print(f"{len(cams)} frames with masks, scene scale {scene_scale:.2f}", flush=True)
    ply_of = {o["id"]: o["ply"] for o in scene["objects"]}
    # Thresholds are set in metres: world metres per reconstruction unit from the manifest.
    unit = float(abs(np.linalg.det(np.asarray(scene["world_from_recon"])[:3, :3])) ** (1 / 3))
    claim = CLAIM_DIST_M / unit
    labels = {o: v for o, v in labels.items() if o in ply_of}
    sam = json.loads((root / "ids" / "masks" / "labels.json").read_text())
    labels, folded = fold_labels(labels, sam)
    if folded:
        print("folded into objects:", folded, flush=True)
    oids = list(labels)
    lut = torch.zeros(256, dtype=torch.uint8, device=dev)  # mask value -> object index + 1
    for k, o in enumerate(oids):
        lut[labels[o]] = k + 1
    all_vals = torch.tensor(sorted({v for vs in labels.values() for v in vs}), device=dev)
    struct_vals = torch.tensor(
        [int(k) for k, n in sam.items() if n.split("#")[0] in STRUCTURE],
        dtype=torch.long, device=dev,
    )  # fmt: skip
    report: dict = {"frames": len(cams), "folded": folded, "objects": {}}

    # The cut. The lift vote leaves parts of objects in the background (blender01: most of the
    # coffee table). A background splat whose centre lands on object k in at least half the
    # views that see it while SAM found k goes to k, if it lies within 15 cm of k.
    bg = load_params(root / "objects" / "background.ply", dev)
    with torch.no_grad():
        obj_masks = [grow(torch.isin(m, all_vals), 6) for m in masks]
        n_in, n_free, hits, seen = label_hits(bg["means"], cams, [lut[m.long()] for m in masks],
                                              obj_masks, len(oids))  # fmt: skip
        del obj_masks
        ghost = (n_in > 0) & (n_free.float() < 0.1 * n_in.float())
        ratio = (hits[:, 1:].float() / seen[:, 1:].clamp(min=1)) * (seen[:, 1:] >= 3)
        best, owner = ratio.max(1)
        claimed = best >= 0.5
    # Only splats near an object join it: a wall splat hidden behind it stays out. A splat
    # seen only on objects (ghost) but claimed by none, because a non-scene item such as a
    # bowl covers it in the views, joins the nearest object within reach. The rest stay:
    # they are the wall and floor behind objects, and the inpainted fill below colours them.
    objs = {o: load_params(root / ply_of[o], dev) for o in oids}
    cand = (claimed | ghost).nonzero()[:, 0]
    with torch.no_grad():
        dist = torch.stack([nearest_dist(bg["means"][cand], objs[o]["means"]) for o in oids], 1)
        near_d, near_k = dist.min(1)
        own = owner[cand]
        own_d = dist.gather(1, own[:, None])[:, 0]
        dest = torch.full_like(own, -1)
        by_ghost = ghost[cand] & (near_d < claim)
        dest[by_ghost] = near_k[by_ghost]
        by_ratio = claimed[cand] & (own_d < claim)
        dest[by_ratio] = own[by_ratio]
    gone = torch.zeros_like(ghost)
    moved: dict[str, int] = {}
    seeds = {}
    for k, o in enumerate(oids):
        params = objs.pop(o)
        sel = cand[dest == k]
        gone[sel] = True
        with torch.no_grad():
            for key in list(params.keys()):
                params[key] = torch.nn.Parameter(torch.cat([params[key], bg[key][sel]]))
        moved[o] = len(sel)
        seeds[o] = {key: v.detach().cpu() for key, v in params.items()}
        del params
    # Hidden splats left inside an object's box (above the floor) are object-surface junk
    # that no object took. The inpainted fill would paint them wall-coloured in front of the
    # object, so they go. The wall and floor behind an object lie outside its box and stay.
    m = torch.tensor(scene["world_from_recon"], dtype=torch.float32, device=dev)
    world = bg["means"] @ m[:3, :3].T + m[:3, 3]
    inside = torch.zeros_like(ghost)
    in_box = {}
    for o in oids:
        w = seeds[o]["means"].to(dev) @ m[:3, :3].T + m[:3, 3]
        lo, hi = torch.quantile(w[:: max(1, len(w) // 100_000)], torch.tensor([0.01, 0.99],
                                device=dev), dim=0)  # fmt: skip
        in_box[o] = ((world > lo - 0.02) & (world < hi + 0.02)).all(1) & (world[:, 1] > FLOOR_CLEAR)
        inside |= in_box[o]
    # Items no noun named (blender01's bowl and succulent on the coffee table) are unlabelled
    # in every view, so neither the cut nor the ghost rule takes them, and they stayed in the
    # background: moving the table left a copy of them behind (1 Nov). Items resting on
    # furniture belong to it (D12): an in-box splat that rarely lands on floor or wall joins
    # the object whose box holds it.
    with torch.no_grad():
        share = struct_share(bg["means"], cams, [torch.isin(mk, struct_vals) for mk in masks])
    items = inside & ~gone & (share < ITEM_STRUCT_SHARE)
    joined = 0
    for o in oids:
        sel = (items & in_box[o]).nonzero()[:, 0]
        items[sel] = False
        if not len(sel):
            continue
        gone[sel] = True
        for key in seeds[o]:
            seeds[o][key] = torch.cat([seeds[o][key], bg[key][sel].detach().cpu()])
        moved[o] += len(sel)
        joined += len(sel)
    report["items_joined"] = joined
    print(f"background: {joined} GSs of unnamed items joined the object under them", flush=True)
    junk = (ghost | (best >= 0.25)) & ~gone & inside
    gone |= junk
    report["junk_removed"] = int(junk.sum())
    hidden_kept = (ghost & ~gone)[~gone]
    with torch.no_grad():
        for key in list(bg.keys()):
            bg[key] = torch.nn.Parameter(bg[key][~gone])
    bg_cpu = {key: v.detach().cpu() for key, v in bg.items()}
    del bg
    torch.cuda.empty_cache()
    print(f"background: {int(gone.sum())} GSs out, moved {moved}", flush=True)
    report["moved_from_background"] = moved
    report["hidden_kept"] = int((ghost & ~gone).sum())

    for oid in oids:
        vals = labels[oid]
        vt = torch.tensor(vals, device=dev)
        frames = [i for i, m in enumerate(masks) if int(torch.isin(m, vt).sum()) >= a.min_px]
        params = torch.nn.ParameterDict(
            {key: torch.nn.Parameter(v.to(dev)) for key, v in seeds.pop(oid).items()}
        )
        if not frames:
            print(f"{oid}: no frame with {a.min_px} mask pixels, kept as lifted")
            save_params(params, out / Path(ply_of[oid]).name)
            continue
        cache: dict = {}

        def parts(i, vt=vt, cache=cache):
            if i not in cache:
                m = torch.isin(masks[i], vt)
                # Other SAM items may cover o (another object), so their pixels say nothing
                # about it, and neither do unlabelled ones: an item no noun named (the bowl
                # on blender01's coffee table) is unlabelled, and forcing o transparent on a
                # black target there turned the bowl black and pruned 90% of it (1 Nov).
                # Floor and wall pixels count: o must be transparent there.
                other = ~torch.isin(masks[i], struct_vals) & ~m
                ignore = grow(other, 3) | (grow(m, 2) & ~shrink(m, 2))
                w = (~ignore).float()
                cache[i] = (m, w)
            return cache[i]

        def weight_fn(i):
            m, w = parts(i)
            return w, imgs[i].float() / 255 * m[..., None]

        def alpha_fn(i):
            m, w = parts(i)
            return m.float(), w

        print(f"{oid}: {len(params['means'])} GSs, {len(frames)} frames", flush=True)
        r = train_part(params, frames, cams, weight_fn, a.obj_steps, scene_scale, dev,
                       alpha_target=alpha_fn, alpha_lambda=0.5, log=oid)  # fmt: skip
        save_params(params, out / Path(ply_of[oid]).name)
        report["objects"][oid] = {**r, "from_background": moved[oid]}
        print(f"{oid}: {r}", flush=True)
        del params, cache
        torch.cuda.empty_cache()

    # Background: first the frame outside the objects (structure labels stay in), geometry
    # and colour. Then colour alone, against the frame with the objects inpainted (OpenCV
    # Telea, no learned weights) inside their masks. The wall and floor
    # behind an object were never seen, so moving it would show whatever colour those
    # splats had. Geometry stays frozen in the fill pass: trained on the fill, faint splats
    # in the air would turn opaque floor-coloured floaters.
    fills: dict = {}

    def bg_weight(i):
        obj = grow(torch.isin(masks[i], all_vals), 6)
        return (~obj).float(), imgs[i].float() / 255

    def fill_weight(i):
        obj = grow(torch.isin(masks[i], all_vals), 6)
        if i not in fills:
            fills[i] = inpaint(imgs[i], obj)
        w = torch.where(obj, BG_FILL_WEIGHT, 1.0)
        return w, torch.where(obj[..., None], fills[i], imgs[i]).float() / 255

    params = torch.nn.ParameterDict(
        {key: torch.nn.Parameter(v.to(dev)) for key, v in bg_cpu.items()}
    )
    # Splats in front of an object's visible surface would have hidden it in the capture, so
    # they cannot be room. The joint fit kept them nearly transparent, but the fill below
    # would paint them. Depth comes from the trained objects rendered together.
    objs = [load_params(out / Path(ply_of[o]).name, dev) for o in oids]
    every = {k: torch.cat([o[k] for o in objs]) for k in objs[0].keys()}
    del objs
    with torch.no_grad():
        front = in_front(params["means"], every, cams, margin=FRONT_MARGIN_M / unit)
        for key in list(params.keys()):
            params[key] = torch.nn.Parameter(params[key][~front])
        # Splats only ever seen behind objects keep a flat colour: SH fitted to a few fill
        # views turns saturated from other angles.
        params["shN"][hidden_kept.to(dev)[~front]] = 0
    del every
    torch.cuda.empty_cache()
    report["in_front_removed"] = int(front.sum())
    print(f"background: {int(front.sum())} GSs in front of objects removed", flush=True)
    print(f"background: {len(params['means'])} GSs", flush=True)
    r = train_part(params, list(range(len(cams))), cams, bg_weight, a.bg_steps, scene_scale,
                   dev, log="background")  # fmt: skip
    # Holes: the wall behind a sofa has almost no splats (blender01: ~40 over 1 m2), so a
    # moved sofa showed the room's outside through it. Depth and colour inpainting extends
    # the surrounding wall and floor into the gap before the colour pass.
    if a.holes:
        obj_masks = [grow(torch.isin(m, all_vals), 6) for m in masks]
        means, rgb, rad = hole_points(params, cams, imgs, obj_masks)
        del obj_masks
        add_splats(params, means, rgb, rad)
        report["hole_splats"] = len(means)
        print(f"background: {len(means)} hole splats added", flush=True)
    rf = train_part(params, list(range(len(cams))), cams, fill_weight, a.fill_steps,
                    scene_scale, dev, log="background fill", colour_only=True)  # fmt: skip
    save_params(params, out / "background.ply")
    report["background"] = {**r, "fill_seconds": rf["seconds"]}
    # Two nouns for one thing (blender01: the armchair is sofa#1 and armchair#0) train into
    # two copies, because both label sets cover it. install() hides the smaller copy.
    report["merged"] = merge_duplicates(
        {o: read_vertices(out / Path(ply_of[o]).name) for o in oids}
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "objects"}), flush=True)


def install(root: Path) -> dict:
    """CPU side, after the Modal run: objects/ -> objects_lifted/ (once), the trained PLYs into
    objects/, and fresh bboxes in scene.json. LOD trees and baked planes were built from the
    old PLYs, so their fields are cleared and the viewer reads the PLYs until
    build_lod.py and bake_planes.py run again (coverage/ stays stale as well)."""
    import shutil  # noqa: PLC0415

    from .scene import Scene  # noqa: PLC0415

    trained, objects = root / "objects_trained", root / "objects"
    if not (objects.parent / "objects_lifted").exists():
        shutil.copytree(objects, root / "objects_lifted")
    scene = Scene.load(root / "scene.json")
    m = np.asarray(scene.world_from_recon, dtype=np.float64)
    report = (
        json.loads((trained / "report.json").read_text())
        if (trained / "report.json").exists()
        else {}
    )
    counts = {}
    for obj in scene.objects:
        if obj.id in report.get("merged", {}):  # a second copy of the kept object
            obj.visible = False
            continue
        src = trained / Path(obj.ply).name
        if obj.source != "scan" or not src.exists():
            continue
        shutil.copyfile(src, root / obj.ply)
        v = read_vertices(src)
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
        # Centre percentiles: a few faint splats far out would inflate a min/max box.
        lo, hi = np.percentile(xyz, 0.5, axis=0), np.percentile(xyz, 99.5, axis=0)
        obj.bbox_min, obj.bbox_max = tuple(map(float, lo)), tuple(map(float, hi))
        obj.lod = None
        counts[obj.id] = len(v)
    if (trained / "background.ply").exists():
        shutil.copyfile(trained / "background.ply", objects / "background.ply")
        scene.background_lod = None
        scene.background_ply = "objects/background.ply"
        scene.planes, scene.plane_bake = [], None
        print("planes cleared: run bake_planes.py --force (and build_lod.py) again")
        counts["background"] = len(read_vertices(objects / "background.ply"))
    scene.save(root / "scene.json")
    return counts


if __name__ == "__main__":
    main()
