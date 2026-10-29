"""Lane bake: floor and ceiling Gaussians to opaque textured quads (docs/ARCHITECTURE.md, 12).

planefit   RANSAC plane, least-squares refinement, oriented bounding rectangle
raster     CPU reference orthographic rasterizer (gsplat classic rules)
texture    push-pull hole filling, opaque texture
glb        hand-written glTF 2.0 binary for one textured quad
pipeline   scene-folder stage used by scripts/bake_planes.py
synthetic  test room in the reconstruction frame with known planes
"""
