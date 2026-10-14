"""NEMBA (embedders_backbones) - launcher.

RUN FROM VSCODE / IDE: open this file and press Run (F5). With no arguments it
opens the GUI. With arguments it runs the CLI:

    python main.py --gui                                   # force the GUI
    python main.py --list                                  # list backbones (+ CPU cost)
    python main.py --demo --backbone tcn --speed fast      # synthetic data with ground truth
    python main.py --backbone tcn --input spikes --file spikes.npy \
                   --behavior behaviour.csv --circular head_direction --out out
    python main.py --backbone tst --input fiber_photometry --file photometry.npy --fs 30 --dff
    python main.py --compare pca,tcn,tst,bilstm --input spikes --file spikes.npy --speed fast
    python main.py                                         # no args -> GUI

Input types: spikes | pose | mask | coco_mask | fiber_photometry | timeseries.
Behaviour files (.csv with a header, .npz, .npy) are decoded with blocked
cross-validation and a circular-shift null; a column named ``time`` is used to
resample behaviour onto the embedding time base.
The package is added to the path automatically, so no install is needed.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def build_parser():
    """All command-line options (grouped as in the GUI)."""
    import argparse
    ap = argparse.ArgumentParser(description="NEMBA: self-supervised neural embeddings + decoding")
    ap.add_argument("--gui", action="store_true", help="launch the GUI")
    ap.add_argument("--list", action="store_true", help="list available backbones")
    ap.add_argument("--demo", action="store_true", help="synthetic recording with known behaviour")
    g = ap.add_argument_group("data")
    g.add_argument("--input", default="spikes",
                   choices=["spikes", "pose", "mask", "coco_mask", "fiber_photometry", "timeseries"])
    g.add_argument("--file", default=None, help="data file (.npy/.npz/.csv/.json/.pkl); empty = demo")
    g.add_argument("--bin-size", type=float, default=0.2, help="spike bin size (s)")
    g.add_argument("--time-axis", type=int, default=1)
    g.add_argument("--fs", type=float, default=None,
                   help="sampling rate for fiber photometry or direct mask videos")
    g.add_argument("--dff", action="store_true", help="dF/F for fiber photometry")
    g.add_argument("--mask-size", type=int, default=64, help="square raster size for COCO masks")
    g.add_argument("--mask-channel-key", default=None,
                   help="COCO annotation key for separate mask channels, e.g. track_id")
    g = ap.add_argument_group("model")
    g.add_argument("--backbone", default="tcn")
    g.add_argument("--speed", default="fast", choices=["fast", "balanced", "accurate"],
                   help="CPU budget preset (steps, window, width, time cap)")
    g.add_argument("--objective", default="masked", choices=["masked", "contrastive", "hybrid"])
    g.add_argument("--max-steps", type=int, default=None, help="override the preset step budget")
    g.add_argument("--epochs", type=int, default=None, help="legacy: epochs -> step budget")
    g.add_argument("--time-budget", type=float, default=None,
                   help="wall-clock cap on training (s); 0 = no cap")
    g.add_argument("--states", type=int, default=None,
                   help="number of states K (default: auto, chosen by silhouette)")
    g.add_argument("--smooth", type=float, default=4.0, help="embedding smoothing sigma (frames)")
    g.add_argument("--input-pca", default="auto", help="'auto', 'none' or an integer")
    g.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("behaviour / decoding")
    g.add_argument("--behavior", default=None, help="behaviour file (.csv/.npz/.npy)")
    g.add_argument("--circular", default="", help="comma list of angular variables (radians)")
    g.add_argument("--decode-model", default="linear", choices=["linear", "knn", "logistic"])
    g.add_argument("--n-perm", type=int, default=200, help="circular-shift null size")
    g = ap.add_argument_group("output")
    g.add_argument("--out", default=os.path.join(HERE, "embedder_out"))
    g.add_argument("--gif-method", default="umap", choices=["umap", "tsne", "pca"])
    g.add_argument("--no-gif", action="store_true")
    g.add_argument("--no-html", action="store_true")
    g.add_argument("--compare", default=None,
                   help="comma list of backbones to run + compare (e.g. pca,tcn,tst)")
    return ap


def cli(argv):
    from embedders_backbones import Embedder, list_backbones, label, data as D
    from embedders_backbones.backbones import CPU_COST_HINT

    args = build_parser().parse_args(argv)
    if args.gui:
        from embedders_backbones.gui import launch
        launch(); return
    if args.list:
        for b in list_backbones():
            print(f"  {b:<12s} {label(b):<32s} CPU cost: {CPU_COST_HINT.get(b, 'fast')}")
        return

    # ---- load data (and behaviour) once ----
    behavior, behavior_times = None, None
    circular = [c.strip() for c in args.circular.split(",") if c.strip()]
    bin_size = args.bin_size
    if args.demo or not args.file:
        from embedders_backbones.synthetic import make_dataset
        print("[demo] synthetic recording: 60 neurons, 600 s, behaviour = phase/speed/state")
        ds = make_dataset(seed=args.seed)
        data, input_kind, bin_size = ds["spike_times"], "spikes", ds["dt"]
        behavior, behavior_times = ds["behavior"], ds["times"]
        circular = circular or ["phase"]
    elif args.input == "spikes":
        data, input_kind = D.load_spikes(args.file), "spikes"
    elif args.input == "coco_mask" or (args.input == "mask" and args.file.endswith(".json")):
        data, input_kind = args.file, "coco_mask"
    else:
        data, input_kind = D.load_array(args.file), args.input
    if args.behavior:
        behavior, behavior_times = D.load_behavior(args.behavior)

    input_pca = None if args.input_pca.lower() == "none" else (
        "auto" if args.input_pca == "auto" else int(args.input_pca))
    budget = "preset" if args.time_budget is None else (None if args.time_budget <= 0 else args.time_budget)
    common = dict(bin_size=bin_size, n_states=args.states, speed=args.speed,
                  objective=args.objective, max_steps=args.max_steps, epochs=args.epochs,
                  time_budget_s=budget, smooth_sigma=args.smooth, input_pca=input_pca,
                  seed=args.seed, device="auto")

    # ---- compare several backbones ----
    if args.compare:
        from embedders_backbones import compare_embedders
        bbs = [b.strip() for b in args.compare.split(",") if b.strip()]
        cmp = compare_embedders(data, input=input_kind, backbones=bbs, time_axis=args.time_axis,
                                fs=args.fs, dff=args.dff, mask_resize=(args.mask_size, args.mask_size),
                                mask_channel_key=args.mask_channel_key, behavior=behavior,
                                behavior_times=behavior_times, circular=circular,
                                decode_model=args.decode_model, n_perm=args.n_perm, **common)
        cmp.run(args.out)
        print("\ncomparison metrics:\n", cmp.metrics.round(3).to_string(index=False))
        if cmp.decoding is not None:
            print("\nheld-out decoding (embedding):\n", cmp.decoding_table().round(3).to_string())
        print(f"\noutputs in {args.out}")
        return

    # ---- single backbone ----
    emb = Embedder(backbone=args.backbone, **common)
    if input_kind == "spikes":
        emb.fit_spikes(data)
    elif input_kind == "coco_mask":
        emb.fit_coco_masks(data, resize=(args.mask_size, args.mask_size),
                           channel_key=args.mask_channel_key, fps=args.fs)
    elif input_kind == "fiber_photometry":
        emb.fit_fiber(data, time_axis=args.time_axis, dff=args.dff, fs=args.fs)
    else:
        emb.fit_features(data, time_axis=args.time_axis)
    met = emb.run(args.out, behavior=behavior, behavior_times=behavior_times, circular=circular,
                  animate=not args.no_gif, gif_method=args.gif_method, html=not args.no_html,
                  n_perm=args.n_perm)
    print("\nmetrics:")
    for k, v in met.items():
        print(f"  {k:<22s} {v:.4f}" if isinstance(v, float) else f"  {k:<22s} {v}")
    if emb.decoding is not None:
        cols = ["target", "features", "metric", "score", "null_p95", "p_holm"]
        print("\ndecoding:\n", emb.decoding.table[cols].round(4).to_string(index=False))
    print(f"\noutputs in {args.out}")


def main():
    argv = sys.argv[1:]
    if not argv:                       # double-clicked / Run in IDE with no args -> GUI
        from embedders_backbones.gui import launch
        launch()
    else:
        cli(argv)


if __name__ == "__main__":
    main()
