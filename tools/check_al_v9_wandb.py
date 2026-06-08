"""Query wandb for any sweep runs by tag and summarise best F-metrics.

Defaults to the al_v9 2x2 sweep (tag=sweep_2x2). Override via flags.

Usage:
    /home/christian/anaconda3/envs/HerdNet/bin/python3 tools/check_al_v9_wandb.py
    /home/christian/anaconda3/envs/HerdNet/bin/python3 tools/check_al_v9_wandb.py --tag sweep_2x2
    /home/christian/anaconda3/envs/HerdNet/bin/python3 tools/check_al_v9_wandb.py --project karisu/hn_active_learning --tag sweep_2x2
"""
from __future__ import annotations

import argparse
import sys

import wandb


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="karisu/hn_active_learning")
    p.add_argument("--tag", default="sweep_2x2")
    args = p.parse_args(argv)

    api = wandb.Api()
    runs = list(api.runs(args.project, filters={"tags": args.tag}, order="-created_at"))
    print(f"Found {len(runs)} runs in {args.project} tagged '{args.tag}':\n", flush=True)

    for r in runs:
        summary = dict(r.summary._json_dict)
        state = r.state
        name = r.name
        last_step = summary.get("_step", "?")
        cur_epoch = summary.get("epoch", summary.get("Epoch", "?"))
        print(f"  {name:<28} state={state:<12} step={last_step}  epoch={cur_epoch}", flush=True)

        keys = ["best_f2_score", "best_f2", "best_f1_score", "best_f1",
                "f2_score", "f1_score", "recall", "precision"]
        found = {k: summary[k] for k in keys if k in summary}
        if found:
            kvs = "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in found.items()
            )
            print(f"      summary: {kvs}", flush=True)

        try:
            hist = list(r.scan_history(
                keys=["f2_score", "f1_score", "recall", "precision", "epoch"],
                page_size=100,
            ))
            valid = [h for h in hist if h.get("f2_score") is not None]
            if valid:
                best = max(valid, key=lambda h: h["f2_score"])
                ep = best.get("epoch", "?")
                print(f"      best in history: epoch={ep}  "
                      f"f2={best['f2_score']:.4f}  "
                      f"r={best.get('recall', 0):.4f}  "
                      f"p={best.get('precision', 0):.4f}  "
                      f"f1={best.get('f1_score', 0):.4f}", flush=True)
                last = valid[-1]
                ep = last.get("epoch", "?")
                print(f"      last in history: epoch={ep}  "
                      f"f2={last['f2_score']:.4f}  "
                      f"r={last.get('recall', 0):.4f}  "
                      f"p={last.get('precision', 0):.4f}", flush=True)
        except Exception as e:
            print(f"      history err: {type(e).__name__}: {e}", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
