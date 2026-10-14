"""Regenerate every image used in README.md (run from anywhere).

    python docs/make_readme_images.py            # figures + explorer + GUI screenshots
    python docs/make_readme_images.py --no-gui   # skip the GUI screenshots
    python docs/make_readme_images.py --no-compare   # skip the comparison figures
    python docs/make_readme_images.py --no-single    # skip the single-backbone figures

Everything is computed on the synthetic recording of ``synthetic.make_dataset``
(60 neurons driven by a ring phase, running speed and 3 behavioural states),
so the pictures in the README are reproducible on any machine.
"""

# =========================================================================== #
# PARAMETERS (edit here)                                                      #
# =========================================================================== #
BACKBONE = "tst"                          # backbone shown in the single-model figures
SPEED = "fast"                            # CPU preset used for every fit
COMPARE = ["pca", "tcn", "tst", "bilstm"]  # backbones of the comparison figures
GUI_COMPARE = "pca,tcn,tst"               # backbones of the GUI comparison screenshot
SEED = 0
FIG_DPI = 150                             # README figures (repo stays light)
GIF_FRAMES = 60
SCREEN_W = 1600                           # GUI screenshots are downscaled to this width
GUI_GEOMETRY = "1480x930+30+30"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
EXPLORER_VIEW = "#color=state&t=0.45"     # explorer URL hash used for its screenshot
# =========================================================================== #

import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG = os.path.join(HERE, "images")
sys.path.insert(0, ROOT)

from embedders_backbones import Embedder, compare_embedders, synthetic, viz  # noqa: E402


def copy(src_dir, name, dst):
    """Copy ``src_dir/name`` to docs/images/``dst``."""
    shutil.copyfile(os.path.join(src_dir, name), os.path.join(IMG, dst))
    print("  wrote", dst)


def crop_background(path, pad=16):
    """Trim uniform background rows at the bottom of a screenshot."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    a = np.asarray(img).astype(int)
    bg = a[-1, -1]
    rows = np.flatnonzero((np.abs(a - bg).sum(2) > 24).any(1))
    if rows.size:
        img.crop((0, 0, img.width, min(img.height, rows[-1] + pad))).save(path, optimize=True)


def figures(ds, tmp):
    """Single-backbone outputs, trajectory GIF and explorer screenshot."""
    beh, bt = ds["behavior"], ds["times"]
    emb = Embedder(BACKBONE, speed=SPEED, bin_size=ds["dt"], seed=SEED)
    emb.fit_spikes(ds["spike_times"])
    out = os.path.join(tmp, "single")
    emb.run(out, behavior=beh, behavior_times=bt, circular=["phase"], formats=("png",),
            animate=False, html=True)
    p = BACKBONE
    for src, dst in [(f"{p}_dashboard.png", "fig_dashboard.png"),
                     (f"{p}_embeddings.png", "fig_embeddings.png"),
                     (f"{p}_embeddings_by_phase.png", "fig_embeddings_by_phase.png"),
                     (f"{p}_decoding.png", "fig_decoding.png"),
                     (f"{p}_states.png", "fig_states.png"),
                     (f"{p}_state_dynamics.png", "fig_state_dynamics.png"),
                     (f"{p}_training.png", "fig_training.png"),
                     (f"{p}_topology.png", "fig_topology.png"),
                     (f"{p}_flow_field.png", "fig_flow_field.png"),
                     (f"{p}_recurrence.png", "fig_recurrence.png"),
                     (f"{p}_rastermap.png", "fig_rastermap.png"),
                     (f"{p}_contingency_state.png", "fig_contingency.png")]:
        copy(out, src, dst)
    viz.trajectory_gif(emb.Z, os.path.join(IMG, "trajectory.gif"), method="umap",
                       projection=emb.projection("umap"), states=emb.states,
                       n_frames=GIF_FRAMES, title=f"{BACKBONE} trajectory")
    print("  wrote trajectory.gif")
    html = os.path.join(out, f"{p}_explorer.html")
    if os.path.exists(CHROME):
        url = "file:///" + html.replace("\\", "/") + EXPLORER_VIEW
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--window-size=1400,880", f"--screenshot={os.path.join(IMG, 'explorer.png')}",
                        url], check=False, capture_output=True, timeout=120)
        crop_background(os.path.join(IMG, "explorer.png"))
        print("  wrote explorer.png")
    else:
        print("  (Chrome not found: explorer screenshot skipped)")


def advisor_figures(ds, tmp):
    """Data-profile figure, advice report and a T-PHATE vs PCA projection."""
    from embedders_backbones.advisor import advise, profile_figure
    rec = advise(ds["spike_times"], behavior=ds["behavior"], behavior_times=ds["times"])
    viz.save_figure(profile_figure(rec.profile), os.path.join(IMG, "fig_data_profile"), ("png",))
    with open(os.path.join(IMG, "advice_example.md"), "w", encoding="utf-8") as fh:
        fh.write(rec.report())
    print("  wrote fig_data_profile.png, advice_example.md")
    emb = Embedder("tcn", speed=SPEED, bin_size=ds["dt"], seed=SEED, log=None)
    emb.fit_spikes(ds["spike_times"])
    beh = emb._align(ds["behavior"], ds["times"], None, ["phase"])
    fig = emb.figure(methods=("pca", "tphate"), color="behavior", behavior=beh["phase"],
                     behavior_name="phase", circular=True)
    viz.save_figure(fig, os.path.join(IMG, "fig_tphate"), ("png",))
    fig = emb.figure(methods=("pca", "tphate"), color="behavior", behavior=beh["state"],
                     behavior_name="true state")
    viz.save_figure(fig, os.path.join(IMG, "fig_tphate_states"), ("png",))
    print("  wrote fig_tphate.png, fig_tphate_states.png")


def comparison(ds, tmp):
    """Comparison figures (embeddings grid, state rasters, decoding bars)."""
    cmp = compare_embedders(ds["spike_times"], input="spikes", backbones=COMPARE,
                            behavior=ds["behavior"], behavior_times=ds["times"],
                            circular=["phase"], speed=SPEED, bin_size=ds["dt"], seed=SEED)
    out = os.path.join(tmp, "compare")
    cmp.run(out, formats=("png",))
    for src, dst in [("compare_embeddings.png", "cmp_embeddings.png"),
                     ("compare_state_sequences.png", "cmp_state_sequences.png"),
                     ("compare_decoding_speed.png", "cmp_decoding_speed.png"),
                     ("compare_train_seconds.png", "cmp_train_seconds.png"),
                     ("compare_consistency.png", "cmp_consistency.png")]:
        copy(out, src, dst)


def gui_screenshots(tmp):
    """Drive the GUI (demo data) and capture every tab."""
    from embedders_backbones import gui
    from PIL import Image
    from _wincapture import capture_tk
    gui.enable_dpi_awareness()
    import tkinter as tk

    def pump(root, cond, timeout):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            root.update()
            if cond():
                return True
            time.sleep(0.02)
        return False

    def shoot(root, app, tab, name):
        if root.state() != "normal":           # minimized windows paint nothing
            root.state("normal"); root.deiconify(); root.geometry(GUI_GEOMETRY)
        app.nb.select(tab)
        pump(root, lambda: False, 2.0)
        img = capture_tk(root)              # PrintWindow: immune to covered / sleeping screens
        if img.width < 800:
            raise RuntimeError(f"window not restored (state={root.state()}, size={img.size})")
        if img.width > SCREEN_W:
            img = img.resize((SCREEN_W, round(img.height * SCREEN_W / img.width)), Image.LANCZOS)
        img.save(os.path.join(IMG, name), optimize=True)
        print("  wrote", name)

    root = tk.Tk()
    app = gui.App(root, settings_path=os.path.join(tmp, "gui_settings.json"))
    root.geometry(GUI_GEOMETRY)
    root.attributes("-topmost", True)
    root.update()
    app.var_file.set(""); app.var_beh.set("")
    app.set_backbone(BACKBONE)
    app.var_speed.set(SPEED); app.var_max_steps.set(""); app.var_states.set("auto")
    app.var_out.set(os.path.join(tmp, "gui_run"))
    app.var_gif.set(False); app.var_html.set(True)
    app.analyse()                                   # advisor on the demo data
    if not pump(root, lambda: not app.busy, 600):
        raise RuntimeError("GUI analyse did not finish")
    shoot(root, app, app.tab_advisor, "gui_advisor.png")
    app.set_backbone(BACKBONE)
    app.run()
    if not pump(root, lambda: not app.busy, 900):
        raise RuntimeError("GUI run did not finish")
    for name, tab in (("gui_overview.png", app.panes["overview"]),
                      ("gui_embedding.png", app.panes["embedding"]),
                      ("gui_decoding.png", app.tab_decoding),
                      ("gui_training.png", app.panes["training"]),
                      ("gui_log.png", app.tab_log)):
        shoot(root, app, tab, name)
    app.var_dyn_fig.set("Flow field")               # dynamics tab, flow field figure
    app.nb.select(app.tab_dynamics)
    app._on_dyn_change()
    pump(root, lambda: getattr(app, "_dyn_shown", None) == "Flow field", 300)
    shoot(root, app, app.tab_dynamics, "gui_dynamics.png")
    app.var_compare.set(GUI_COMPARE)
    app.var_max_steps.set("150")
    app.compare()
    if not pump(root, lambda: not app.busy, 1200):
        raise RuntimeError("GUI compare did not finish")
    shoot(root, app, app.tab_compare, "gui_compare.png")
    root.destroy()


def main():
    os.makedirs(IMG, exist_ok=True)
    viz.STYLE["dpi"] = FIG_DPI
    ds = synthetic.make_dataset(seed=SEED)
    with tempfile.TemporaryDirectory() as tmp:
        if "--no-single" not in sys.argv:
            print("single-backbone figures ..."); figures(ds, tmp)
        if "--no-advisor" not in sys.argv:
            print("advisor figures ..."); advisor_figures(ds, tmp)
        if "--no-compare" not in sys.argv:
            print("comparison figures ..."); comparison(ds, tmp)
        if "--no-gui" not in sys.argv:
            print("GUI screenshots ..."); gui_screenshots(tmp)
    print(f"images in {IMG}")


if __name__ == "__main__":
    main()
