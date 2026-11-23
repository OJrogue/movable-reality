"""Limits for the publish kit: title, description, tags, X posts, chapters, subtitles, files."""
import re
import subprocess
from pathlib import Path

P = Path("data/broll/publish")
s = (P / "PUBLISH.md").read_text(encoding="utf8")


def section(a, b):
    return s[s.index(a) + len(a):s.index(b, s.index(a))].strip()


title = section("### Title", "### Description")
desc = section("### Description", "### Tags")
tags = section("### Tags", "### Settings")
post = section("### Post", "### Reply under it")
reply = section("### Reply under it", "### After posting")
x_len = lambda t: len(re.sub(r"https?://\S+", "x" * 23, t))
print("title", len(title), "/100")
print("description", len(desc), "/5000")
print("tags", len(tags), "/500")
print("post", x_len(post), "/280")
print("reply", x_len(reply), "/280")
print("semicolons", s.count(";"))
print("chapters", re.findall(r"^(\d:\d\d) ", desc, re.M))
srt = (P / "movable_reality.en.srt").read_text(encoding="utf8")
print("srt cues", len(re.findall(r"-->", srt)), "longest line", max(len(x) for x in srt.splitlines()))
for f in ("movable_reality_final.mp4", "movable_reality_x_captions.mp4"):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,pix_fmt", "-of", "csv=p=0", str(P / f)],
                       capture_output=True, encoding="utf8").stdout.split()
    print(f, round((P / f).stat().st_size / 2**20, 1), "MiB", r)
print("thumbnail KiB", round((P / "youtube_thumbnail.jpg").stat().st_size / 1024))
