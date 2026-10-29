"""Coverage loop (lane: coverage): score how well the capture observed the room, find gaps,
rank next-best views for top-up videos, repair small gaps with Muse Image, tag generated
splats.

Modules:
  cameras     pinhole cameras in the OpenCV convention, look-at, frame changes
  colmap      read and write COLMAP binary models (MapAnything export layout)
  ply         3DGS PLY reading and writing, per-splat `generated` property
  geometry    per-Gaussian view counts (frustum + depth test) and angle spread
  voxels      voxel grid, occupancy, ray marching (seen free space, VR visibility)
  gaps        combined scores, connected components, Gap entries
  nbv         walkable standing viewpoints and next-best-view ranking
  uncertainty U-3DGS-style per-primitive photometric uncertainty (pure math)
  pipeline    score a scene folder end to end (used by scripts/score_coverage.py)
  synthetic   synthetic room capture with known hidden regions (ground truth)
  evaluate    precision and recall of gaps against the synthetic ground truth
  repair      Muse Image edits of rendered gap views (cached)
"""
