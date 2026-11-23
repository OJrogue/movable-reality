"""Frame-by-frame scenes of the second cut: pipeline animation (real blender07 data) and centred slides."""
import json
import math
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

C = Path("data/broll/_cut")
OUT = C / "frames"
SC = "data/scenes/blender07/"
W, H, FPS = 1920, 1080, 30
BG = (15, 16, 21)
FG = (228, 225, 233)
DIM = (150, 150, 165)
PRI = (190, 198, 255)
CARD = (27, 28, 36)
F = "C:/Windows/Fonts/"
_fonts = {}


def font(size, kind="b"):
    k = (size, kind)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(F + {"b": "segoeuib.ttf", "r": "segoeui.ttf", "s": "seguisb.ttf", "m": "consola.ttf"}[kind], size)
    return _fonts[k]


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def back(u):
    u = min(1.0, max(0.0, u))
    return 1 + 2.2 * (u - 1) ** 3 + 1.2 * (u - 1) ** 2


PAL = [(255, 107, 107), (255, 179, 71), (255, 225, 86), (120, 224, 143), (72, 209, 204), (92, 170, 255), (170, 130, 255), (255, 130, 200),
       (180, 240, 90), (255, 150, 110), (110, 230, 230), (210, 170, 255)]


def pill(d, xy, text, f, fill, fg=FG, pad=(16, 7), anchor="lt"):
    x, y = xy
    b = d.textbbox((0, 0), text, font=f)
    w, h = b[2] - b[0] + pad[0] * 2, b[3] - b[1] + pad[1] * 2
    if anchor == "mm":
        x, y = x - w / 2, y - h / 2
    d.rounded_rectangle([x, y, x + w, y + h], radius=h / 2, fill=fill)
    d.text((x + pad[0] - b[0], y + pad[1] - b[1]), text, font=f, fill=fg)
    return w, h


def ctext(d, y, text, f, fill=FG, x=W / 2):
    d.text((x, y), text, font=f, fill=fill, anchor="mm")


# ---------- pipeline ----------
RAIL = [("Video", ""), ("Muse Wonder", "lists objects"), ("SAM 3.1", "tracks them"), ("MapAnything", "finds cameras"),
        ("Gaussian splatting", "builds the room"), ("Objects", "each its own splats")]
BEATS = [0.0, 3.1, 6.0, 8.3, 10.5, 13.75, 17.4]  # seconds, set to the voice takes
_d = {}


def data():
    if not _d:
        z = np.load(C / "anim_data.npz")
        xyz, rgb, oidx, size = z["xyz"], z["rgb"], z["oidx"], z["size"]
        keep = (xyz[:, 0] > -1.3) & (xyz[:, 0] < 4.7) & (xyz[:, 2] > -5.2) & (xyz[:, 2] < 1.95) & (xyz[:, 1] > -0.15) & (xyz[:, 1] < 2.3)
        xyz, rgb, oidx, size = xyz[keep], rgb[keep], oidx[keep], size[keep]
        # dollhouse: open the two walls towards the camera
        wall = (oidx == 0) & ((xyz[:, 0] > 4.15) | (xyz[:, 2] > 1.45))
        xyz, rgb, oidx, size = xyz[~wall], rgb[~wall], oidx[~wall], size[~wall]
        _d.update(xyz=xyz, rgb=rgb, oidx=oidx, size=size, cams=z["cams"], names=[str(n) for n in z["names"]])
        c = np.zeros((len(_d["names"]) + 1, 3))
        for i in range(1, len(c)):
            s = xyz[oidx == i]
            c[i] = np.median(s, 0) if len(s) else 0
        _d["centre"] = c
        _d["count"] = np.bincount(oidx, minlength=len(c))
    return _d


def view(az_deg, el_deg=38.0, r=7.4):
    t = np.array([1.75, 0.55, -1.7])
    az, el = math.radians(az_deg), math.radians(el_deg)
    e = t + r * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    f = (t - e) / np.linalg.norm(t - e)
    s = np.cross(f, [0, 1, 0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return e, s, u, f


FOCAL = 1120.0
CY = 640


def project(p, cam):
    e, s, u, f = cam
    d = p - e
    z = d @ f
    return W / 2 + FOCAL * (d @ s) / z, CY - FOCAL * (d @ u) / z, z


def points(img, p, rgb, cam, a=None, size=None):
    x, y, z = project(p, cam)
    if size is not None:  # large splats first, as blocks, so flat areas fill in
        for lim, r in ((0.05, 9), (0.02, 5)):
            b = size > lim
            o = np.argsort(-z[b])
            xb, yb = x[b][o].astype(int), y[b][o].astype(int)
            cb = (rgb[b][o] * 255).astype(np.uint8)
            m = (xb > r) & (xb < W - r) & (yb > r) & (yb < H - r)
            xb, yb, cb = xb[m], yb[m], cb[m]
            for dx in range(-r // 2, r // 2 + 1, 2):
                for dy in range(-r // 2, r // 2 + 1, 2):
                    img[yb + dy, xb + dx] = cb
                    img[yb + dy + 1, xb + dx + 1] = cb
    o = np.argsort(-z)
    x, y, z = x[o], y[o], z[o]
    c = (rgb[o] * 255).astype(np.uint8)
    if a is not None:
        c = (c * a[o][:, None]).astype(np.uint8)
    xi, yi = x.astype(int), y.astype(int)
    m = (xi > 0) & (xi < W - 3) & (yi > 0) & (yi < H - 3)
    xi, yi, c = xi[m], yi[m], c[m]
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)):
        img[yi + dy, xi + dx] = c
    return img


def grid(d, cam):
    for v in np.arange(-1, 5.01, 1.0):
        a = project(np.array([[v, 0, -5.0]]), cam)
        b = project(np.array([[v, 0, 1.75]]), cam)
        d.line([a[0][0], a[1][0], b[0][0], b[1][0]], fill=(62, 65, 82), width=1)
    for v in np.arange(-5, 1.76, 1.0):
        a = project(np.array([[-1.0, 0, v]]), cam)
        b = project(np.array([[4.4, 0, v]]), cam)
        d.line([a[0][0], a[1][0], b[0][0], b[1][0]], fill=(62, 65, 82), width=1)


def frustum(d, c2w, cam, col, size=0.085):
    R, o = c2w[:3], c2w[3]
    corners = [o + size * (R[:, 0] * sx * 1.5 + R[:, 1] * sy + R[:, 2] * 1.4) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    pts = np.array([o] + corners)
    x, y, _ = project(pts, cam)
    q = list(zip(x.tolist(), y.tolist()))
    for i in range(1, 5):
        d.line([q[0], q[i]], fill=col, width=1)
    d.line(q[1:] + [q[1]], fill=col, width=2)


def rail(d, t):
    n = len(RAIL)
    x0, x1, y = 220, W - 220, 74
    step = (x1 - x0) / (n - 1)
    # progress: node k lights when beat k-1 starts (video is on at once)
    prog = 0.0
    for k in range(len(BEATS) - 2):
        prog += ease((t - BEATS[k]) / 0.5)
    prog = min(prog, n - 1)
    d.line([x0, y, x1, y], fill=(52, 54, 68), width=4)
    d.line([x0, y, x0 + step * prog, y], fill=PRI, width=4)
    for i, (a, b) in enumerate(RAIL):
        x = x0 + step * i
        on = prog >= i - 0.01
        cur = on and prog < i + 0.99
        r = 12 if cur else 8
        d.ellipse([x - r, y - r, x + r, y + r], fill=PRI if on else (52, 54, 68))
        d.text((x, y + 36), a, font=font(25, "b"), fill=FG if on else DIM, anchor="mm")
        if b:
            d.text((x, y + 66), b, font=font(20, "r"), fill=(FG if cur else DIM) if on else (90, 92, 105), anchor="mm")


def best_window():
    p = C / "best_window.txt"
    if p.exists():
        return int(p.read_text())
    n = len(list(Path(SC + "ids/masks").glob("*.png")))
    counts = []
    for i in range(0, n, 10):
        q = Path(SC + f"ids/masks/{i:06d}.png")
        counts.append(len(np.unique(np.array(Image.open(q)))) if q.exists() else 0)
    w = 18
    s = np.convolve(counts, np.ones(w), "valid")
    best = int(np.argmax(s)) * 10
    p.write_text(str(best))
    return best


INV = None
LBL = None


def frame_box(idx, mask_a, t_local):
    """Input frame with SAM masks over it. Returns an RGB image of 1120x630."""
    global LBL
    if LBL is None:
        LBL = json.loads(Path(SC + "ids/labels.json").read_text(encoding="utf8"))
    im = Image.open(SC + f"frames/{idx:06d}.jpg").convert("RGB").resize((1120, 630), Image.BILINEAR)
    if mask_a <= 0:
        return im
    q = Path(SC + f"ids/masks/{idx:06d}.png")
    if not q.exists():
        return im
    mk = np.array(Image.open(q).resize((1120, 630), Image.NEAREST)).astype(int)
    a = np.array(im).astype(float)
    for v in np.unique(mk):
        if v == 0:
            continue
        m = mk == v
        c = np.array(PAL[v % len(PAL)], dtype=float)
        a[m] = a[m] * (1 - 0.5 * mask_a) + c * 0.5 * mask_a
    edge = (mk != np.roll(mk, 1, 0)) | (mk != np.roll(mk, 1, 1))
    a[edge & (mk > 0)] = 255
    im = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(im)
    if mask_a > 0.6:
        for v in np.unique(mk):
            if v == 0 or (mk == v).sum() < 7000:
                continue
            ys, xs = np.nonzero(mk == v)
            name = LBL.get(str(v - 1), LBL.get(str(v), "")).split("#")[0]
            if name:
                pill(d, (float(np.median(xs)), float(np.median(ys))), name, font(20, "s"), tuple(int(x * 0.55) for x in PAL[v % len(PAL)]), anchor="mm", pad=(12, 5))
    return im


def pipeline(i):
    global INV
    t = i / FPS
    if INV is None:
        INV = [o["noun"] for o in json.loads(Path(SC + "inventory.json").read_text(encoding="utf8"))["objects"]]
    img = Image.new("RGB", (W, H), BG)
    if t < BEATS[2]:
        idx = best_window() + i
        ma = ease((t - BEATS[1]) / 0.5)
        fb = frame_box(idx, ma, t)
        x0, y0 = 150, 240
        ImageDraw.Draw(fb)
        img.paste(fb, (x0, y0))
        d = ImageDraw.Draw(img)
        x, y = x0 + 1120 + 44, y0
        d.text((x, y), "Objects found", font=font(30, "b"), fill=FG)
        y += 56
        xx = x
        for k, noun in enumerate(INV[:24]):
            s = ease((t - 0.25 - k * 0.095) / 0.18)
            if s <= 0:
                continue
            b = d.textbbox((0, 0), noun, font=font(22, "s"))
            w = b[2] - b[0] + 28
            if xx + w > W - 50:
                xx, y = x, y + 50
            col = tuple(int(c * 0.5) for c in PAL[(k + 1) % len(PAL)]) if ma > 0.5 else (46, 48, 62)
            pill(d, (xx, y), noun, font(22, "s"), col, pad=(14, 7))
            xx += w + 10
    else:
        p = data()
        az = 128 + (t - BEATS[2]) * 4.2
        cam = view(az)
        arr = np.array(img)
        n_all = len(p["xyz"])
        if t < BEATS[3]:
            pil = Image.fromarray(arr)
            d = ImageDraw.Draw(pil)
            grid(d, cam)
            u = (t - BEATS[2]) / (BEATS[3] - BEATS[2] - 0.3)
            cams = p["cams"]
            show = int(min(1, u) * len(cams))
            for c in range(0, show, 5):
                near = c > show - 40
                frustum(d, cams[c], cam, (255, 225, 110) if near else (140, 150, 220))
            img = pil
        elif t < BEATS[5]:
            u = ease((t - BEATS[3]) / (BEATS[4] - BEATS[3] - 0.2))
            n = int(n_all * (0.015 + 0.985 * u ** 1.6))
            arr = points(arr, p["xyz"][:n], p["rgb"][:n], cam, size=p["size"][:n])
            img = Image.fromarray(arr)
            d = ImageDraw.Draw(img, "RGBA")
            if t < BEATS[4] + 0.2:
                fade = int(200 * (1 - u))
                cams = p["cams"]
                for c in range(0, len(cams), 6):
                    frustum(d, cams[c], cam, (140, 150, 220, fade))
            if t >= BEATS[4]:
                a = ease((t - BEATS[4]) / 0.25)
                terminal(img, t - BEATS[4], a)
        else:
            e = t - BEATS[5]
            xyz = p["xyz"].copy()
            ox = p["oidx"]
            c = p["centre"]
            room = np.array([1.75, 0, -1.7])
            lift = np.zeros((len(c), 3))
            for k in range(1, len(c)):
                s = back((e - 0.05 - (k % 12) * 0.05) / 0.9)
                lift[k] = s * (np.array([c[k][0] - room[0], 0, c[k][2] - room[2]]) * 0.14 + np.array([0, 0.62 + 0.12 * (k % 3), 0]))
            xyz = xyz + lift[ox]
            bgm = ox == 0
            col = p["rgb"].copy()
            col[bgm] *= 1 - 0.55 * ease(e / 0.6)
            arr = points(arr, xyz, col, cam, size=p["size"])
            img = Image.fromarray(arr)
            d = ImageDraw.Draw(img, "RGBA")
            order = [k + 1 for k, n in enumerate(p["names"]) if n.split("#")[0] in ("sofa", "armchair", "bookshelf", "coffee table", "tv stand", "ottoman", "side table") and n[-2:] in ("#0", "#1")]
            for k in order:
                a = ease((e - 0.8) / 0.3)
                if a <= 0:
                    continue
                x, y, _ = project((c[k] + lift[k]).reshape(1, 3) + np.array([[0, 0.25, 0]]), cam)
                name = p["names"][k - 1].split("#")[0]
                pill(d, (float(x[0]), float(y[0])), name, font(21, "s"), (24, 26, 40, int(225 * a)), fg=(*PRI, int(255 * a)), anchor="mm", pad=(12, 5))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 158], fill=BG)
    rail(d, t)
    # lead into the next scene
    img.save(OUT / "pipeline" / f"{i:05d}.jpg", quality=93)


STAGES = ["frames", "inventory", "reconstruct", "train splats", "lift object ids", "train objects", "complete with SAM 3D", "coverage", "levels of detail"]


def terminal(img, e, a):
    w, h = 1120, 440
    x, y = (W - w) // 2, 330
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rounded_rectangle([x, y, x + w, y + h], 18, fill=(10, 11, 16, int(235 * a)), outline=(70, 72, 92, int(255 * a)), width=2)
    cmd = "uv run python scripts/run_pipeline.py walk.mp4 --scene room"
    n = int(len(cmd) * min(1, e / 0.9))
    d.text((x + 32, y + 28), "$ " + cmd[:n], font=font(28, "m"), fill=(*FG, int(255 * a)))
    for k, s in enumerate(STAGES):
        u = (e - 0.95 - k * 0.2)
        if u <= 0:
            continue
        col = (120, 224, 143, 255) if u > 0.2 else (*DIM, 255)
        tick = "✓" if u > 0.2 else "…"
        d.text((x + 32, y + 86 + k * 36), tick, font=ImageFont.truetype(F + "seguisym.ttf", 26), fill=col)
        d.text((x + 70, y + 88 + k * 36), s, font=font(26, "r"), fill=(*FG, 255))
    img.paste(ov, (0, 0), ov)


# ---------- input walk with chips ----------
def jpg_seq(clip, name, start, dur, vf=""):
    d = C / "src" / name
    if d.exists() and any(d.iterdir()):
        return d
    d.mkdir(parents=True, exist_ok=True)
    f = "fps=30" + ("," + vf if vf else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(dur), "-i", str(C.parent / f"{clip}.mp4"), "-vf", f, "-q:v", "3", str(d / "%05d.jpg")], check=True)
    return d


def walk(i):
    t = i / FPS
    src = C / "src" / "walk"
    fs = sorted(src.glob("*.jpg"))
    img = Image.open(fs[min(i, len(fs) - 1)]).convert("RGB")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for k, (txt, at) in enumerate((("No depth sensor", 4.2), ("No floor plan", 5.5), ("No manual labelling", 7.05))):
        s = back((t - at) / 0.22)
        if t < at:
            continue
        a = min(1, (t - at) / 0.12)
        y = 330 + k * 120 + (1 - s) * 30
        b = d.textbbox((0, 0), txt, font=font(60, "b"))
        w = b[2] - b[0] + 72
        d.rounded_rectangle([(W - w) / 2, y, (W + w) / 2, y + 100], 50, fill=(14, 15, 22, int(225 * a)))
        d.text((W / 2, y + 50), txt, font=font(60, "b"), fill=(*FG, int(255 * a)), anchor="mm")
    b = "A video walkthrough + the ceiling height"
    if t < 4.2:
        a = ease(t / 0.3) * (1 - ease((t - 3.9) / 0.3))
        d.rounded_rectangle([W / 2 - 560, 930, W / 2 + 560, 1020], 45, fill=(14, 15, 22, int(215 * a)))
        d.text((W / 2, 975), b, font=font(46, "s"), fill=(*FG, int(255 * a)), anchor="mm")
    img.paste(ov, (0, 0), ov)
    img.save(OUT / "walk" / f"{i:05d}.jpg", quality=93)


# ---------- centred slides over the blurred room ----------
def blurred(i):
    fs = sorted((C / "src" / "glide").glob("*.jpg"))
    im = Image.open(fs[i % len(fs)]).convert("RGB").resize((480, 270), Image.BILINEAR).filter(ImageFilter.GaussianBlur(9)).resize((W, H), Image.BILINEAR)
    return Image.blend(im, Image.new("RGB", (W, H), BG), 0.8)


def grad_text(img, xy, text, f, stops=((196, 202, 255), (255, 214, 190)), a=1.0):
    d = ImageDraw.Draw(img)
    b = d.textbbox((0, 0), text, font=f)
    w, h = b[2] - b[0], b[3] - b[1]
    m = Image.new("L", (w + 8, h + 8), 0)
    ImageDraw.Draw(m).text((4 - b[0], 4 - b[1]), text, font=f, fill=int(255 * a))
    g = np.linspace(0, 1, w + 8)[None, :, None]
    col = (np.array(stops[0])[None, None] * (1 - g) + np.array(stops[1])[None, None] * g).astype(np.uint8)
    col = np.repeat(col, h + 8, 0)
    img.paste(Image.fromarray(col), (int(xy[0] - w / 2), int(xy[1] - h / 2)), m)


def demo(i):
    t = i / FPS
    img = blurred(i)
    a = ease(t / 0.3)
    dy = (1 - back(t / 0.35)) * 26
    d = ImageDraw.Draw(img)
    ctext(d, 400 + dy, "A working demo,", font(104, "b"), tuple(int(x * a) for x in FG))
    if t > 0.4:
        pass
    a2 = ease((t - 0.25) / 0.3)
    if a2 > 0:
        grad_text(img, (W / 2, 525 + (1 - a2) * 20), "and a good place to start.", font(104, "b"), a=a2)
    for k, (txt, at) in enumerate((("Two rooms rendered in Blender", 1.55), ("A start for a project or research", 3.5))):
        s = ease((t - at) / 0.25)
        if s > 0:
            x = W / 2 + (-330 if k == 0 else 330)
            pill(ImageDraw.Draw(img), (x, 690 + (1 - s) * 18), txt, font(32, "s"), (40, 44, 68), fg=tuple(int(c * s) for c in FG), anchor="mm", pad=(26, 14))
    img.save(OUT / "demo" / f"{i:05d}.jpg", quality=93)


_el = None


def models(i):
    global _el
    t = i / FPS
    if _el is None:
        _el = (Image.open(C / "el_models.png").convert("RGB"), json.loads((C / "el_models.json").read_text()))
    el, r = _el
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    sc = 1680 / r["w"]
    gw, gh = 1680, int(r["h"] * sc)
    gx, gy = (W - gw) // 2, (H - gh - 150) // 2 + 70
    a = ease(t / 0.25)
    ctext(d, gy - 62, "Eight Meta models", font(64, "b"), tuple(int(c * a) for c in FG))
    big = el.resize((gw, gh), Image.LANCZOS)
    d.rounded_rectangle([gx, gy, gx + gw, gy + gh], 26, fill=(21, 22, 29))
    for k, (x, y, w, h) in enumerate(r["cells"]):
        s = ease((t - 0.1 - k * 0.13) / 0.2)
        if s <= 0:
            continue
        box = tuple(int(v * sc) for v in (x, y, x + w, y + h))
        cell = big.crop(box)
        if s < 1:
            cell = Image.blend(Image.new("RGB", cell.size, (21, 22, 29)), cell, s)
        img.paste(cell, (gx + box[0], gy + box[1] + int((1 - s) * 14)))
    for k, (txt, at) in enumerate((("MIT licensed", 2.05), ("Each stage is a script you can swap", 3.9), ("Test rooms and scoring tools included", 6.9))):
        s = back((t - at) / 0.25)
        if t < at:
            continue
        al = min(1, (t - at) / 0.15)
        y = gy + gh + 62 + k * 0
        x = {0: 330, 1: 830, 2: 1460}[k]
        d = ImageDraw.Draw(img)
        pill(d, (x, y + (1 - s) * 18), txt, font(33, "s"), (46, 50, 80), fg=tuple(int(c * al) for c in FG), anchor="mm", pad=(24, 14))
    img.save(OUT / "models" / f"{i:05d}.jpg", quality=93)


def end(i):
    t = i / FPS
    img = blurred(i + 200)
    d = ImageDraw.Draw(img)
    a = ease(t / 0.3)
    ctext(d, 370, "Film a room.", font(116, "b"), tuple(int(c * a) for c in FG))
    grad_text(img, (W / 2, 505), "Move the furniture.", font(116, "b"), a=ease((t - 0.2) / 0.3))
    s = ease((t - 0.5) / 0.3)
    if s > 0:
        d = ImageDraw.Draw(img)
        pill(d, (W / 2, 670), "ojytcanal--movable-reality-demo.us-east.modal.run", font(38, "s"), (58, 63, 110), fg=tuple(int(c * s) for c in FG), anchor="mm", pad=(30, 16))
        ctext(d, 770, "Movable Reality · Team Larracat · MIT licensed", font(30, "r"), tuple(int(c * s) for c in DIM))
    img.save(OUT / "end" / f"{i:05d}.jpg", quality=93)


JOBS = {"pipeline": (pipeline, BEATS[-1]), "walk": (walk, 8.9), "demo": (demo, 6.7), "models": (models, 11.7), "end": (end, 4.2)}


def run(name):
    fn, dur = JOBS[name]
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    n = int(dur * FPS)
    with Pool(6) as pool:
        for _ in pool.imap_unordered(fn, range(n), chunksize=8):
            pass
    print("done", name, n)


if __name__ == "__main__":
    names = sys.argv[1:] or list(JOBS)
    if names[0] == "still":
        OUT.joinpath(names[1]).mkdir(parents=True, exist_ok=True)
        for t in names[2:]:
            JOBS[names[1]][0](int(float(t) * FPS))
        sys.exit(0)
    jpg_seq("01_input_walk_8x", "walk", 0, 9)
    jpg_seq("15_room_glide", "glide", 2, 20)
    for n in names:
        run(n)
