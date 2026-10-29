# Bake lane screenshots (PR evidence)

Screenshots for the `bake:` pull request, kept on this orphan branch so no images land on
`main`. Produced 29 Oct 2026 in headless Chromium 1.63 (Playwright) with
`--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`.

- `depth_*.png`, `depth_grid.png`: `web/src/planes/depthcheck.html?case=<case>`
  (quad, splats, behind, front, both, crossing). Red = opaque three.js quad at z = -2,
  green = splats at z = -3, blue = splats at z = -1 (or tilted from -1 to -3 for crossing).
- `viewer_grid.jpg`: `?scene=bakedemo` (synthetic room from `scripts/bake_planes.py
  --synthetic`), splats only vs baked planes, before and after moving the sofa 1.3 m.
- `floor_compare.jpg`: floor crop, splats (left) vs baked quad (right).
