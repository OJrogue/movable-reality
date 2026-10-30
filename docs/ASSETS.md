# Third-party assets

Assets used to build the **synthetic** ground-truth rooms `blender01` and `blender02`
(`scripts/blender_room.py`). The room is a benchmark and a fallback demo scene; the real
phone capture stays the main demo. All assets are CC0 (public domain) from
[Poly Haven](https://polyhaven.com/license). None are committed: `scripts/blender_assets.py`
downloads them into `data/assets/` (gitignored) and writes `data/assets/assets.json` with
the same list. The room shell (floor, walls, ceiling geometry), the floor lamp, the lights
and the camera paths are built in `scripts/blender_room.py` during the event.

| Asset | Kind (res.) | Author(s) | Licence | Role in `blender01` |
|---|---|---|---|---|
| [Sofa 02](https://polyhaven.com/a/sofa_02) | model (2k) | Kirill Sannikov | CC0 | sofa, back to the north wall (its back is never filmed) |
| [Arm Chair 01](https://polyhaven.com/a/ArmChair_01) | model (2k) | Kirill Sannikov | CC0 | armchair |
| [Modern Coffee Table 01](https://polyhaven.com/a/modern_coffee_table_01) | model (2k) | Amin | CC0 | coffee table |
| [Side Table 01](https://polyhaven.com/a/side_table_01) | model (1k) | James Ray Cock | CC0 | side table |
| [Painted Wooden Cabinet](https://polyhaven.com/a/painted_wooden_cabinet) | model (2k) | Kirill Sannikov | CC0 | cabinet, back to the south wall |
| [Potted Plant 01](https://polyhaven.com/a/potted_plant_01) | model (2k) | Rico Cilliers | CC0 | potted plant |
| [Book Encyclopedia Set 01](https://polyhaven.com/a/book_encyclopedia_set_01) | model (1k) | John Malcolm | CC0 | books on the cabinet |
| [Ceramic Vase 01](https://polyhaven.com/a/ceramic_vase_01) | model (1k) | James Ray Cock | CC0 | vase on the side table |
| [Potted Plant 04](https://polyhaven.com/a/potted_plant_04) | model (1k) | James Ray Cock | CC0 | succulent on the coffee table |
| [Wooden Bowl 01](https://polyhaven.com/a/wooden_bowl_01) | model (1k) | Oliver Harries | CC0 | bowl on the coffee table |
| [Laminate Floor 02](https://polyhaven.com/a/laminate_floor_02) | texture (2k) | Charlotte Baglioni, Dario Barresi | CC0 | floor |
| [Beige Wall 001](https://polyhaven.com/a/beige_wall_001) | texture (2k) | Dimitrios Savva, Rico Cilliers | CC0 | walls |

Added for `blender02`, the L-shaped bedroom-study (30 Oct). It also reuses Book
Encyclopedia Set 01 (books on the desk), Potted Plant 04 (succulent on the desk), both
textures and the built floor lamp.

| Asset | Kind (res.) | Author(s) | Licence | Role in `blender02` |
|---|---|---|---|---|
| [Vintage Day Bed](https://polyhaven.com/a/vintage_day_bed) | model (2k) | Aron Łyczek | CC0 | bed, back to the north wall of the alcove |
| [Painted Wooden Nightstand](https://polyhaven.com/a/painted_wooden_nightstand) | model (1k) | Kirill Sannikov | CC0 | nightstand next to the bed |
| [Wooden Table 02](https://polyhaven.com/a/wooden_table_02) | model (2k) | Serhii Khromov | CC0 | desk against the south wall |
| [Dining Chair 02](https://polyhaven.com/a/dining_chair_02) | model (2k) | James Ray Cock | CC0 | desk chair |
| [Wooden Bookshelf Worn](https://polyhaven.com/a/wooden_bookshelf_worn) | model (2k) | Ulan Cabanilla | CC0 | bookshelf against the east wall |
| [Modern Arm Chair 01](https://polyhaven.com/a/modern_arm_chair_01) | model (2k) | Vibrant Nordic | CC0 | armchair in the south-east corner |
| [Potted Plant 02](https://polyhaven.com/a/potted_plant_02) | model (2k) | Rico Cilliers | CC0 | potted plant by the west wall |
| [Ceramic Vase 03](https://polyhaven.com/a/ceramic_vase_03) | model (1k) | James Ray Cock | CC0 | vase on the nightstand |
| [Horse Statue 01](https://polyhaven.com/a/horse_statue_01) | model (1k) | Rico Cilliers | CC0 | horse figurine on the nightstand |

Tool: Blender 4.5 LTS (GPL-3.0) renders the room. It is a tool only; nothing from Blender
ships with the project. Cycles' denoiser, Intel OpenImageDenoise (Apache-2.0), is a
learned network. It cleans the offline renders that make the test video; no product
stage calls it (allowed by OJ, decision D19 in `docs/HANDOFF.md`).
`--samples 128 --no-denoise` renders without it.
