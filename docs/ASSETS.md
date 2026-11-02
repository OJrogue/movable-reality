# Third-party assets

Assets used to build the **synthetic** ground-truth rooms `blender01` to `blender05`
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

`blender03` (living room with an angled wall, a bay window and a free-standing column) and
`blender04` (attic bedroom-study with sloped ceilings and a dormer, 31 Oct) add no new
assets, and neither does `blender05` (`blender01` with windows and doors, 31 Oct). They
reuse models listed above in new places. The room shells, including the column, the
slopes, the dormer, the windows and the doors, are built in `scripts/blender_room.py`.

| Scene | Furniture (asset) | Items on furniture |
|---|---|---|
| `blender03` | sofa (Sofa 02), coffee table (Modern Coffee Table 01), side table (Side Table 01), floor lamp (built), cabinet (Painted Wooden Cabinet), potted plant (Potted Plant 01), bookshelf (Wooden Bookshelf Worn), armchair in the bay (Modern Arm Chair 01) | books, vase (Ceramic Vase 01), succulent, bowl |
| `blender04` | bed (Vintage Day Bed) and nightstand under the west slope, floor lamp (built), bookshelf on the north gable, armchair (Arm Chair 01) and potted plant (Potted Plant 02) under the east slope, desk (Wooden Table 02) and chair (Dining Chair 02) in the dormer | vase (Ceramic Vase 03), horse figurine, books, succulent |
| `blender05` | the `blender01` furniture in the same places, plus two windows (north wall above the sofa, east wall) with casing, sill and closed venetian blinds, and two closed panelled doors with lever handles (west and south walls), all built in `scripts/blender_room.py` with flat paint materials (31 Oct) | the `blender01` items |

Tool: Blender 4.5 LTS (GPL-3.0) renders the room (Blender 5.2.1 on the Mac). It is a tool only; nothing from Blender
ships with the project. Cycles' denoiser, Intel OpenImageDenoise (Apache-2.0), is a
learned network. It cleans the offline renders that make the test video; no product
stage calls it (allowed by OJ, decision D19 in `docs/HANDOFF.md`).
`--samples 128 --no-denoise` renders without it.

## `blender06`, the realistic bedroom (2 Nov)

Built by `scripts/blender_bedroom.py`, meant as the main synthetic test room. New assets,
all downloaded by `scripts/blender_assets.py`. It also reuses Painted Wooden Nightstand,
Potted Plant 02, Potted Plant 04 and Book Encyclopedia Set 01 from the tables above.

| Asset | Kind (res.) | Author(s) | Licence | Role in `blender06` |
|---|---|---|---|---|
| [Painted Wooden Chair 01](https://polyhaven.com/a/painted_wooden_chair_01) | model (2k) | Kuutti Siitonen | CC0 | desk chair |
| [Desk Lamp Arm 01](https://polyhaven.com/a/desk_lamp_arm_01) | model (2k) | Kuutti Siitonen, Yann Kervran | CC0 | desk lamp |
| [Alarm Clock 01](https://polyhaven.com/a/alarm_clock_01) | model (1k) | James Ray Cock, Yann Kervran | CC0 | alarm clock on the nightstand |
| [Carved Wooden Elephant](https://polyhaven.com/a/carved_wooden_elephant) | model (1k) | Greg Zaal | CC0 | elephant on the chest of drawers |
| [Ukulele 01](https://polyhaven.com/a/Ukulele_01) | model (2k) | Joseph Burgan | CC0 | ukulele leaning on the east wall |
| [Wicker Basket 02](https://polyhaven.com/a/wicker_basket_02) | model (2k) | Kuutti Siitonen | CC0 | laundry basket |
| [Modern Ceiling Lamp 01](https://polyhaven.com/a/modern_ceiling_lamp_01) | model (1k) | James Ray Cock | CC0 | pendant lamp (scaled 0.7, unlit by default) |
| [Wall Clock](https://polyhaven.com/a/wall_clock) | model (1k) | PierreB3D | CC0 | wall clock above the chest of drawers |
| [Standing Picture Frame 01](https://polyhaven.com/a/standing_picture_frame_01) | model (1k) | James Ray Cock | CC0 | photo frame on the wall shelf (its glass faces removed) |
| [Wood Floor](https://polyhaven.com/a/wood_floor) | texture (2k) | Dimitrios Savva | CC0 | floor |
| [White Stucco](https://polyhaven.com/a/white_stucco) | texture (2k) | Amal Kumar | CC0 | walls (warm white, sage-green accent wall) and ceiling |
| [White Oak Veneer](https://polyhaven.com/a/white_oak_veneer) | texture (2k) | Jenelle van Heerden | CC0 | bed frame, desk top, shelf, chest of drawers |
| [Poly Wool Herringbone](https://polyhaven.com/a/poly_wool_herringbone) | texture (2k) | Rico Cilliers, colormass | CC0 | headboard, rug |
| [Rough Linen](https://polyhaven.com/a/rough_linen) | texture (2k) | Rico Cilliers, colormass | CC0 | duvet, pillow case, curtains |
| [Waffle Pique Cotton](https://polyhaven.com/a/waffle_pique_cotton) | texture (2k) | Rico Cilliers, colormass | CC0 | throw at the foot of the bed |
| [Cotton Jersey](https://polyhaven.com/a/cotton_jersey) | texture (2k) | Rico Cilliers, colormass | CC0 | fitted sheet, sweater on the chair |
| [Eilenriede Park](https://polyhaven.com/a/eilenriede_park) | HDRI (4k) | Greg Zaal | CC0 | the park outside the window and the daylight |
| [The Great Wave off Kanagawa](https://commons.wikimedia.org/wiki/File:Tsunami_by_hokusai_19th_century.jpg) | image (1920 px) | Katsushika Hokusai, Wikimedia Commons | Public domain | framed print above the bed |
| [Earthrise](https://commons.wikimedia.org/wiki/File:NASA-Apollo8-Dec24-Earthrise.jpg) | image (1920 px) | NASA / Bill Anders (Apollo 8), Wikimedia Commons | Public domain | framed print above the desk |

Built in `scripts/blender_bedroom.py` (no third-party geometry): the room shell with the
window hole and reveal, the casement window, door, radiator, skirting, switch and sockets,
the bed (frame, upholstered headboard, mattress, pillow; duvet, throw and sweater draped
with Blender's cloth solver), the chest of drawers, desk, wall shelf, books, laptop (its
screen image is drawn by the script), mouse, mug, notebook and pen, waste
bin, backpack, rug, curtains and rod, the tin robot and the two action figures.
