"""embedders_backbones - launcher.

RUN FROM VSCODE / IDE: open this file and press Run (F5). With no arguments it
opens the small GUI. With arguments it runs the CLI:

    python main.py --gui                                  # force the GUI
    python main.py --backbone dtc --input spikes --file spikes.npy --out out
    python main.py --backbone tst --input fiber_photometry --file photometry.npy --fs 30
    python main.py --list                                 # list backbones
    python main.py                                        # no args -> GUI

Input types: spikes | pose | mask | fiber_photometry | timeseries.
Add the package to the path automatically, so no install is needed.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def cli(argv):
    import argparse
    from embedders_backbones import Embedder, list_backbones, data as D

    ap = argparse.ArgumentParser(description="embedders_backbones CLI")
    ap.add_argument("--gui", action="store_true", help="launch the GUI")
    ap.add_argument("--list", action="store_true", help="list available backbones")
    ap.add_argument("--backbone", default="dtc")
    ap.add_argument("--input", default="spikes",
                    choices=["spikes", "pose", "mask", "fiber_photometry", "timeseries"])
    ap.add_argument("--file", default=None, help="data file (.npy/.npz/.csv/.pkl); empty = demo")
    ap.add_argument("--out", default=os.path.join(HERE, "embedder_out"))
    ap.add_argument("--states", type=int, default=None,
                    help="number of states K (default: auto, chosen by silhouette)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bin-size", type=float, default=0.2)
    ap.add_argument("--time-axis", type=int, default=1)
    ap.add_argument("--fs", type=float, default=None, help="sampling rate (fiber photometry)")
    ap.add_argument("--dff", action="store_true", help="dF/F for fiber photometry")
    ap.add_argument("--gif-method", default="umap", choices=["umap", "tsne", "pca"])
    ap.add_argument("--compare", default=None,
                    help="comma list of backbones to run + compare (e.g. dtc,tst,bilstm)")
    args = ap.parse_args(argv)

    if args.gui:
        from embedders_backbones.gui import launch; launch(); return
    if args.list:
        for b in list_backbones():
            from embedders_backbones import label; print(f"  {b:<12s} {label(b)}")
        return

    # ---- load data once ----
    if not args.file:
        from embedders_backbones.gui import _demo_spikes
        print("[no --file] running the synthetic demo")
        data, input_kind = _demo_spikes(), "spikes"
    elif args.input == "spikes":
        data, input_kind = D.load_spikes(args.file), "spikes"
    else:
        data, input_kind = D.load_array(args.file), args.input

    # ---- compare several backbones ----
    if args.compare:
        from embedders_backbones import compare_embedders
        bbs = [b.strip() for b in args.compare.split(",") if b.strip()]
        cmp = compare_embedders(data, input=input_kind, backbones=bbs, time_axis=args.time_axis,
                                fs=args.fs, dff=args.dff, n_states=args.states,
                                epochs=args.epochs, device="auto")
        cmp.run(args.out)
        print("\ncomparison metrics:\n", cmp.metrics.round(3).to_string(index=False))
        print(f"\nbest by silhouette: {cmp.best('silhouette')}\noutputs in {args.out}")
        return

    # ---- single backbone ----
    emb = Embedder(backbone=args.backbone, bin_size=args.bin_size, n_states=args.states,
                   epochs=args.epochs, device="auto")
    if input_kind == "spikes":
        emb.fit_spikes(data)
    elif input_kind == "fiber_photometry":
        emb.fit_fiber(data, time_axis=args.time_axis, dff=args.dff, fs=args.fs)
    else:
        emb.fit_features(data, time_axis=args.time_axis)
    met = emb.run(args.out, gif_method=args.gif_method)
    print("\nmetrics:")
    for k, v in met.items():
        print(f"  {k:<20s} {v:.4f}" if isinstance(v, float) else f"  {k:<20s} {v}")
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
