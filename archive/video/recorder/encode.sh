#!/bin/bash
# raw/*.webm -> data/broll/*.mp4: 1920x1080, constant 30 fps, H.264, no audio.
# Agent clips also get a _fast cut: real time while typing and for the reply, 10x while waiting.
R="$(dirname "$0")/raw"
D=/d/Projects/MetaHackathon/movable-reality/data/broll
mkdir -p "$D"
enc=(-c:v libx264 -preset medium -crf 17 -pix_fmt yuv420p -movflags +faststart -an)
vf="fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
one() { # raw name, output name
  [ -f "$R/$1.webm" ] || { echo "missing $1"; return; }
  ffmpeg -v error -y -fflags +genpts -i "$R/$1.webm" -vf "$vf" "${enc[@]}" "$D/$2.mp4" && echo "$2"
}
fast() { # output name, seconds typed, seconds of the reply at the end
  local dur; dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$D/$1.mp4")
  local b; b=$(python3 -c "print(max($2, $dur-$3))" 2>/dev/null || awk "BEGIN{print $dur-$3}")
  ffmpeg -v error -y -i "$D/$1.mp4" -filter_complex \
    "[0]trim=0:$2,setpts=PTS-STARTPTS[a];[0]trim=$2:$b,setpts=(PTS-STARTPTS)/10[b];[0]trim=$b,setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3,fps=30" \
    "${enc[@]}" "$D/$1_fast.mp4" && echo "$1_fast"
}
one drag_armchair 03_drag_armchair
one gizmo_move 04_gizmo_move
one gizmo_rotate 05_gizmo_rotate
one measure 07_measure
one library_add 08_library_add
one agent_move 09_agent_move
one agent_fit 10_agent_fit
one agent_add 11_agent_add
one voice 13_voice
one gaps 14_coverage_gaps
one glide 15_room_glide
one room_before 16_room_before
one room_after 17_room_after
for c in 09_agent_move 11_agent_add 13_voice; do [ -f "$D/$c.mp4" ] && fast $c 7 6; done
# Landing page and loading: screencast frames with their real timing.
if [ -f "$R/landing_frames/list.txt" ]; then
  (cd "$R/landing_frames" && ffmpeg -v error -y -f concat -safe 0 -i list.txt -vf "$vf" "${enc[@]}" "$D/02_landing_and_loading.mp4") && echo landing
fi
# The input video: 122 s walk at 8x, with no overlay.
V=/d/Projects/MetaHackathon/movable-reality/data/raw/blender07.mp4
ffmpeg -v error -y -i "$V" -vf "setpts=PTS/8,fps=30,scale=1920:1080" "${enc[@]}" "$D/01_input_walk_8x.mp4" && echo walk
ls -la --time-style=+ "$D"
