"""Lane lod: per-object splat PLYs and Spark LOD trees (.RAD) for the viewer.

- `split`: split a trained scene PLY into background and per-object PLYs by Gaussian id.
- `rad`: read the header of a Spark .RAD file (no dependency on Spark).
- `build`: find or set up Spark's `build-lod` tool and build one .RAD per object PLY.

See docs/LOD.md for how Spark uses these trees at runtime.
"""
