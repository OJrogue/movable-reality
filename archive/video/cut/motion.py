"""When things change in the agent clips: frame difference over the sofa region, per 0.2 s."""
import subprocess

import numpy as np

W, H = 480, 270
for name, box in (("09_agent_move_fast", (150, 60, 330, 170)), ("11_agent_add_fast", (300, 50, 400, 180))):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"data/broll/{name}.mp4", "-vf", f"fps=5,scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(float)
    x0, y0, x1, y1 = box
    d = np.abs(np.diff(a[:, y0:y1, x0:x1], axis=0)).mean((1, 2))
    print(name, len(a), "top:", [(round((int(i) + 1) / 5, 1), round(float(d[i]), 2)) for i in np.argsort(-d)[:8]])
