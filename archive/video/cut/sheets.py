"""Timed contact sheets of the clips, and one sheet of the rendered slides."""
import subprocess

B = "data/broll/"
for name, step in (("08_library_add", 1), ("13_voice_fast", 0.8), ("09_agent_move_fast", 1.2), ("10_agent_fit", 1.1), ("11_agent_add_fast", 0.7)):
    vf = f"fps=1/{step},scale=480:-1,tile=6x4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", B + name + ".mp4", "-vf", vf, "-frames:v", "1", B + f"_cut/s_{name}.jpg"], check=True)
F = B + "_cut/frames/"
inputs = []
for p in ("walk/00060", "walk/00250", "demo/00180", "models/00300", "end/00100", "pipeline/00495"):
    inputs += ["-i", F + p + ".jpg"]
subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", "xstack=inputs=6:grid=3x2,scale=1900:-1", "-frames:v", "1", B + "_cut/slide_sheet.jpg"], check=True)
print("ok")
