"""The room (background splats) as one vertex-coloured triangle mesh, for the mesh-only view.

    python -m movable_reality.roommesh --scene_dir /data/scenes/<id> [--voxel_m 0.015]

The background splats are rendered from every reconstruction camera (gsplat, colour and
expected depth). The depth maps go into a truncated signed distance volume (open3d
ScalableTSDFVolume, a fusion algorithm without learned weights) with the rendered colour,
and marching cubes turns the volume into a mesh. Only pixels with alpha of at least
`--alpha` count, so faint floaters leave no surface. Expected depth also mixes a floater
with the wall behind it into a depth in mid-air, which the volume never carves away (the
first run on blender01 was full of such sheets). So a depth pixel counts only when at
least `--min_agree` of its `--neighbours` nearest views see a surface at the same depth
(within `--agree` of it, relative), the usual consistency check of multi-view depth
fusion. Connected pieces smaller than `--min_tris` triangles go, and quadric decimation
brings the mesh down to `--faces`.

Input is the background before any plane bake (plane_bake.source_background_ply, else
background_ply), so the floor is part of the mesh. Cameras are recon_full/ (the frame of
the PLYs). Writes room/room_mesh.ply (recon frame, sRGB vertex colours) and
room/report.json. Needs CUDA and gsplat (the Modal image). `modal_jobs/reconstruct.py
--room-mesh` runs it on Modal and converts the result to room/room_mesh.glb for the viewer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .coverage.colmap import read_model


def unit_m(world_from_recon) -> float:
    """World metres per reconstruction unit (cube root of the determinant)."""
    return float(abs(np.linalg.det(np.asarray(world_from_recon)[:3, :3])) ** (1 / 3))


def source_background(scene: dict) -> str:
    """The background PLY with the floor and ceiling splats still in it."""
    bake = scene.get("plane_bake") or {}
    return bake.get("source_background_ply") or scene.get("background_ply")


def consistent(views: list[dict], centres: np.ndarray, k: int, tol: float, dev) -> list:
    """Per view, the number of neighbour views whose own depth agrees with each pixel's
    point (|z_j - d_j| <= tol * z_j at the pixel it lands on). Neighbours are the k views
    with the nearest centres that face a similar way (optical axes within 75 deg)."""
    import torch  # noqa: PLC0415

    axes = np.array([v["vm"][2, :3] for v in views])  # camera +Z in world (row 2 of R)
    out = []
    for i, vi in enumerate(views):
        d = np.linalg.norm(centres - centres[i], axis=1)
        d[(axes @ axes[i]) < np.cos(np.radians(75))] = np.inf
        d[i] = np.inf
        nbr = [j for j in np.argsort(d)[:k] if np.isfinite(d[j])]
        H, W = vi["H"], vi["W"]
        z = vi["depth"]
        vs, us = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                                indexing="ij")  # fmt: skip
        Ki = torch.tensor(vi["K"], dtype=torch.float32, device=dev)
        x = (us + 0.5 - Ki[0, 2]) / Ki[0, 0] * z
        y = (vs + 0.5 - Ki[1, 2]) / Ki[1, 1] * z
        pc = torch.stack([x, y, z], -1).reshape(-1, 3)
        inv = torch.tensor(np.linalg.inv(vi["vm"]), dtype=torch.float32, device=dev)
        pw = pc @ inv[:3, :3].T + inv[:3, 3]
        count = torch.zeros(H * W, dtype=torch.int32, device=dev)
        for j in nbr:
            vj = views[j]
            M = torch.tensor(vj["vm"], dtype=torch.float32, device=dev)
            Kj = torch.tensor(vj["K"], dtype=torch.float32, device=dev)
            q = pw @ M[:3, :3].T + M[:3, 3]
            zj = q[:, 2]
            u = (q[:, 0] / zj.clamp(min=1e-6) * Kj[0, 0] + Kj[0, 2]).floor().long()
            v = (q[:, 1] / zj.clamp(min=1e-6) * Kj[1, 1] + Kj[1, 2]).floor().long()
            inside = (zj > 1e-4) & (u >= 0) & (u < vj["W"]) & (v >= 0) & (v < vj["H"])
            dj = torch.zeros_like(zj)
            dj[inside] = vj["depth"][v[inside], u[inside]]
            count += (inside & (dj > 0) & ((zj - dj).abs() <= tol * zj)).int()
        out.append((count * (pc[:, 2] > 0)).reshape(H, W))
    return out


def main() -> None:
    import open3d as o3d  # noqa: PLC0415 - Modal image
    import torch  # noqa: PLC0415
    from gsplat import rasterization  # noqa: PLC0415

    from .objtrain import load_params  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=Path, required=True)
    ap.add_argument("--voxel_m", type=float, default=0.015)
    ap.add_argument("--trunc_voxels", type=float, default=4.0)
    ap.add_argument("--scale", type=float, default=0.5, help="render scale of the frames")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--min_tris", type=int, default=2000)
    ap.add_argument("--faces", type=int, default=800_000)
    ap.add_argument("--neighbours", type=int, default=12)
    ap.add_argument("--min_agree", type=int, default=3)
    ap.add_argument("--agree", type=float, default=0.01, help="relative depth tolerance")
    a = ap.parse_args()
    t0 = time.time()
    root, dev = a.scene_dir, "cuda"
    scene = json.loads((root / "scene.json").read_text())
    unit = unit_m(scene["world_from_recon"])
    voxel = a.voxel_m / unit
    bg_rel = source_background(scene)
    params = load_params(root / bg_rel, dev)
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    cams = read_model(root / "recon_full")
    centres = np.array([c.center for c in cams])
    far = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 4 + 10 * a.voxel_m / unit
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=a.trunc_voxels * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )  # fmt: skip
    kept, views = [], []
    with torch.no_grad():
        for c in cams:
            W, H = round(c.width * a.scale), round(c.height * a.scale)
            K = np.asarray(c.K, dtype=np.float64).copy()
            K[:2] *= np.array([W / c.width, H / c.height])[:, None]
            vm = np.eye(4)
            vm[:3, :3], vm[:3, 3] = c.R, c.t
            out, alpha, _ = rasterization(
                params["means"], params["quats"], torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]), colors,
                torch.tensor(vm, dtype=torch.float32, device=dev)[None],
                torch.tensor(K, dtype=torch.float32, device=dev)[None], W, H, sh_degree=3,
                render_mode="RGB+ED", packed=False, rasterize_mode="antialiased",
            )  # fmt: skip
            depth = out[0, ..., 3].float()
            depth = torch.where((alpha[0, ..., 0] >= a.alpha) & (depth > 0), depth, 0)
            rgb = (out[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
            views.append({"vm": vm, "K": K, "W": W, "H": H, "depth": depth, "rgb": rgb})
        agree = consistent(views, centres, a.neighbours, a.agree, dev)
        for v, n_ok in zip(views, agree, strict=True):
            ok = n_ok >= a.min_agree
            depth = torch.where(ok, v["depth"], 0).cpu().numpy().astype(np.float32)
            kept.append(float(ok.float().mean()))
            rgb, W, H, K, vm = v["rgb"], v["W"], v["H"], v["K"], v["vm"]
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(depth),
                depth_scale=1.0, depth_trunc=far, convert_rgb_to_intensity=False,
            )  # fmt: skip
            intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            vol.integrate(rgbd, intr, vm)
    t_fuse = time.time() - t0
    mesh = vol.extract_triangle_mesh()
    raw = len(mesh.triangles)
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[clusters] < a.min_tris)
    mesh.remove_unreferenced_vertices()
    cleaned = len(mesh.triangles)
    if cleaned > a.faces:
        mesh = mesh.simplify_quadric_decimation(a.faces)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    out_dir = root / "room"
    out_dir.mkdir(exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_dir / "room_mesh.ply"), mesh, write_ascii=False)
    report = {
        "background_ply": bg_rel,
        "cameras": len(cams),
        "render_scale": a.scale,
        "alpha_min": a.alpha,
        "mean_kept_share": round(float(np.mean(kept)), 3),
        "neighbours": a.neighbours,
        "min_agree": a.min_agree,
        "agree": a.agree,
        "voxel_m": a.voxel_m,
        "faces_raw": raw,
        "faces_after_small_pieces": cleaned,
        "small_pieces_removed": int((counts < a.min_tris).sum()),
        "faces": len(mesh.triangles),
        "vertices": len(mesh.vertices),
        "fuse_s": round(t_fuse, 1),
        "total_s": round(time.time() - t0, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
