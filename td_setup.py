# td_setup.py — build the BLR Metro AQI ingestion network inside td.toe.
#
# Recommended (handles paste/blank-line issues automatically):
#     exec(open('/path/to/td_setup.py').read())
#
# Or, paste the whole file in Textport (Alt-T). Either way it's idempotent —
# re-running tears down and rebuilds /project1/aqi_animation.
#
# After running, look inside /project1/aqi_animation:
#     strip_<set>   Movie File In TOP  — 6×4 sprite sheet (downloaded once)
#     frame_<set>   Crop TOP           — current hour 1024×973 LA frame
#     out_<set>     Null TOP           — stable downstream tap
#
# L channel = normalised value (0 → vmin, 1 → vmax). Alpha = city mask.

BASE_URL     = "https://diagram-chasing.github.io/metro-aqi-exhibition/aqi"
GRID_COLS    = 6   # must match build_strips.py
GRID_ROWS    = 4
LOOP_SECONDS = 24  # seconds per full 24h loop

# vmin/vmax baked from each set's manifest.json. Refresh if you regenerate
# strips and the data range shifts (or wire a Web Client DAT for live read).
SETS = {
    "hourly":  {"vmin": 8.898, "vmax": 44.486, "unit": "ug/m3"},
    "avoided": {"vmin": 0.000, "vmax":  0.250, "unit": "reduction"},
    "net":     {"vmin": 9.454, "vmax": 46.384, "unit": "ug/m3"},
}


def build():
    parent_comp = op("/project1") or root

    # Wipe any prior build so re-running is idempotent.
    old = parent_comp.op("aqi_animation")
    if old is not None:
        old.destroy()

    container = parent_comp.create(containerCOMP, "aqi_animation")
    container.nodeX, container.nodeY = 0, 0

    hour_chop = container.create(constantCHOP, "hour_clock")
    hour_chop.nodeX, hour_chop.nodeY = -900, 200
    hour_chop.par.name0 = "hour"
    hour_chop.par.value0.expr = f"(absTime.seconds % {LOOP_SECONDS}) / {LOOP_SECONDS} * 24"
    hour_chop.par.value0.mode = ParMode.EXPRESSION

    hour_expr = "int(op('hour_clock')['hour'])"

    for i, (name, rng) in enumerate(SETS.items()):
        y = -i * 260

        strip = container.create(moviefileinTOP, f"strip_{name}")
        strip.par.file = f"{BASE_URL}/{name}/strip.png"
        strip.par.play = 0
        strip.par.index = 0
        strip.par.reloadpulse.pulse()
        strip.nodeX, strip.nodeY = -600, y

        crop = container.create(cropTOP, f"frame_{name}")
        crop.setInputs([strip])
        # Crop TOP's cropleft/right/bottom/top are already 0–1 fractions.
        crop.par.cropleft.expr   = f"({hour_expr} % {GRID_COLS}) / {GRID_COLS}"
        crop.par.cropright.expr  = f"(({hour_expr} % {GRID_COLS}) + 1) / {GRID_COLS}"
        crop.par.cropbottom.expr = f"({hour_expr} // {GRID_COLS}) / {GRID_ROWS}"
        crop.par.croptop.expr    = f"(({hour_expr} // {GRID_COLS}) + 1) / {GRID_ROWS}"
        for p in ("cropleft", "cropright", "cropbottom", "croptop"):
            getattr(crop.par, p).mode = ParMode.EXPRESSION
        crop.nodeX, crop.nodeY = -300, y

        out = container.create(nullTOP, f"out_{name}")
        out.setInputs([crop])
        out.nodeX, out.nodeY = 0, y
        out.viewer = True
        out.comment = (
            f"{name}: L = (value - {rng['vmin']}) / ({rng['vmax']} - {rng['vmin']})  "
            f"alpha = city mask  |  unit = {rng['unit']}"
        )

    # Single combined preview: hourly | avoided | net, side-by-side.
    outs = [container.op(f"out_{n}") for n in SETS]
    final = container.create(layoutTOP, "final")
    final.setInputs(outs)
    final.nodeX, final.nodeY = 350, -260
    final.viewer = True
    final.comment = "Layout: out_hourly | out_avoided | out_net"
    # Layout TOP defaults to horizontal arrangement; adjust 'align' in the UI
    # if you want vertical or grid.

    container.allowCooking = True
    container.openViewer(unique=True)
    print("Built /project1/aqi_animation with strip_/frame_/out_ for:",
          ", ".join(SETS.keys()))
    print("Preview: /project1/aqi_animation/final  (3-up layout)")
    print("To wallpaper it: right-click the network background, choose")
    print("  'Wallpaper Settings' and set TOP = aqi_animation/final.")


build()
