from __future__ import annotations
import argparse
from pathlib import Path

from .train import train_model
from .evaluate import evaluate as run_evaluate
from .visualization import (
    plot_data_summary, plot_training, plot_predictions, plot_nse_map,
    plot_flow_duration, plot_metrics_box, plot_skill_vs_attrs,
)


def _add_train(sp):
    p = sp.add_parser("train", help="train a sequence model")
    p.add_argument("--model", default="lstm", choices=["lstm"])
    p.add_argument("--seq-len", type=int, default=365)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--loss", choices=["nse", "mse"], default="nse")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--patience", type=int, default=8, help="early-stop patience on val NSE")
    p.add_argument("--min-delta", type=float, default=1e-4, help="min val NSE gain to count as improvement")
    return p


def _add_eval(sp):
    p = sp.add_parser("evaluate", help="evaluate a checkpoint on the test set")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default="outputs")
    return p


def _add_plot(sp):
    p = sp.add_parser("plot", help="produce figures from a checkpoint + metrics")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--split", default="test")
    return p


def _add_summarize(sp):
    p = sp.add_parser("summarize-data", help="dataset summary plots and tables")
    p.add_argument("--out", default="outputs/figs/summary")
    return p


def _build_parser():
    parser = argparse.ArgumentParser(prog="camelflow", description="camelflow — LSTM streamflow prediction on minicamels")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_summarize(sub)
    _add_train(sub)
    _add_eval(sub)
    _add_plot(sub)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.cmd == "summarize-data":
        plot_data_summary(args.out)
        return

    if args.cmd == "train":
        cfg = vars(args).copy()
        cfg.pop("cmd")
        train_model(cfg)
        return

    if args.cmd == "evaluate":
        run_evaluate(args.checkpoint, out_dir=args.out, split=args.split)
        return

    if args.cmd == "plot":
        out = Path(args.out)
        history_path = out / "history.json"
        if history_path.exists():
            plot_training(str(history_path), out_dir=str(out / "figs" / "train"))
        run_evaluate(args.checkpoint, out_dir=str(out), split=args.split)
        metrics_csv = out / f"{args.split}_metrics.csv"
        preds_npz = out / f"{args.split}_predictions.npz"
        pred_dir = str(out / "figs" / "pred")
        plot_predictions(str(preds_npz), str(metrics_csv), out_dir=pred_dir)
        plot_flow_duration(str(preds_npz), str(metrics_csv), out_dir=pred_dir)
        plot_metrics_box(str(metrics_csv), out_dir=pred_dir)
        plot_skill_vs_attrs(str(metrics_csv), out_dir=pred_dir)
        plot_nse_map(str(metrics_csv), out_dir=pred_dir)
        return


if __name__ == "__main__":
    main()
