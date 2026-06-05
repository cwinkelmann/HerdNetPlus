#!/usr/bin/env python3
"""Find and rank the best training runs across all output directories.

Scans output/**/best_model.pth for stored metrics, or falls back to parsing
training log files for [BEST_METRICS] / [SUMMARY] lines.

Usage:
    python tools/best_runs.py                          # scan ./output/
    python tools/best_runs.py --output-dir ./output/   # explicit dir
    python tools/best_runs.py --sort-by f2_score       # rank by F2
    python tools/best_runs.py --top 5                  # show top 5
    python tools/best_runs.py --log /tmp/comparison.log # parse a log file
    python tools/best_runs.py --csv results.csv        # export to CSV
"""

import argparse
import csv as csv_module
import re
import sys
from pathlib import Path

import torch


def load_metrics_from_pth(pth_path: Path) -> dict | None:
    """Extract metrics dict from a best_model.pth file."""
    try:
        pth = torch.load(pth_path, weights_only=False, map_location='cpu')
    except Exception:
        return None

    result = {'path': str(pth_path.parent)}

    # Extract model name from config
    if 'config' in pth:
        cfg = pth['config']
        if isinstance(cfg, dict):
            result['model'] = cfg.get('model', {}).get('name', '?')
            result['epochs'] = cfg.get('training_settings', {}).get('epochs', '?')
            result['wandb_run'] = cfg.get('wandb_run', '?')
        else:
            # OmegaConf DictConfig
            try:
                result['model'] = cfg.model.name
                result['epochs'] = cfg.training_settings.epochs
                result['wandb_run'] = cfg.get('wandb_run', '?')
            except Exception:
                result['model'] = '?'

    result['epoch'] = pth.get('epoch', '?')

    # Primary: use stored metrics dict (new format)
    if 'metrics' in pth and isinstance(pth['metrics'], dict):
        result.update(pth['metrics'])
        return result

    # Fallback: use best_val only
    if 'best_val' in pth:
        result['f1_score'] = pth['best_val']
        return result

    return result


def parse_metrics_from_log(log_path: str) -> list[dict]:
    """Parse [BEST_METRICS] and [SUMMARY] lines from a log file."""
    results = []

    with open(log_path) as f:
        log = f.read()

    # Parse [SUMMARY] lines (one per training run)
    for m in re.finditer(
        r'\[SUMMARY\] '
        r'output_dir=(\S+) '
        r'model=(\S+) '
        r'best_f1=([\d.]+) '
        r'best_f2=([\d.]+) '
        r'recall=([\d.]+) '
        r'precision=([\d.]+) '
        r'mae=([\d.]+) '
        r'rmse=([\d.]+) '
        r'best_val=([\d.]+) '
        r'epochs=(\d+)',
        log
    ):
        results.append({
            'path': m.group(1),
            'model': m.group(2),
            'f1_score': float(m.group(3)),
            'f2_score': float(m.group(4)),
            'recall': float(m.group(5)),
            'precision': float(m.group(6)),
            'mae': float(m.group(7)),
            'rmse': float(m.group(8)),
            'best_val': float(m.group(9)),
            'epochs': int(m.group(10)),
        })

    if results:
        return results

    # Fallback: parse [BEST_METRICS] lines (last one per run is the final best)
    # Group by output directory using "Best model by End User Metric" lines
    best_saves = list(re.finditer(
        r'Best model by End User Metric \w+ saved - Epoch (\d+) - '
        r'Validation value: ([\d.]+), path: (.+?)/best_model\.pth',
        log
    ))

    best_metrics = list(re.finditer(
        r'\[BEST_METRICS\] - Epoch: \[(\d+)\] '
        r'f1=([\d.]+) f2=([\d.]+) f5=([\d.]+) '
        r'recall=([\d.]+) precision=([\d.]+) '
        r'mae=([\d.]+) me=([-\d.]+) rmse=([\d.]+) '
        r'tp=(\d+) fn=(\d+) fp=(\d+) '
        r'avg_score=([\d.]+)',
        log
    ))

    # Use the last BEST_METRICS per output dir
    seen_dirs = {}
    for bm in best_metrics:
        epoch = int(bm.group(1))
        entry = {
            'epoch': epoch,
            'f1_score': float(bm.group(2)),
            'f2_score': float(bm.group(3)),
            'f5_score': float(bm.group(4)),
            'recall': float(bm.group(5)),
            'precision': float(bm.group(6)),
            'mae': float(bm.group(7)),
            'me': float(bm.group(8)),
            'rmse': float(bm.group(9)),
            'tp': int(bm.group(10)),
            'fn': int(bm.group(11)),
            'fp': int(bm.group(12)),
            'avg_score': float(bm.group(13)),
        }
        # Find matching save line to get the path
        for bs in best_saves:
            if int(bs.group(1)) == epoch and abs(float(bs.group(2)) - entry['f1_score']) < 0.001:
                entry['path'] = bs.group(3)
                break
        seen_dirs[entry.get('path', f'unknown_{epoch}')] = entry

    if seen_dirs:
        return list(seen_dirs.values())

    # Last fallback: parse old-format "Best model" lines for F1 only
    for m in best_saves:
        path = m.group(3)
        entry = {
            'path': path,
            'epoch': int(m.group(1)),
            'f1_score': float(m.group(2)),
        }
        # Keep only the last (best) per path
        seen_dirs[path] = entry

    return list(seen_dirs.values())


def scan_output_dir(output_dir: Path) -> list[dict]:
    """Scan all best_model.pth files in output directory tree."""
    results = []
    for pth_path in sorted(output_dir.rglob('best_model.pth')):
        entry = load_metrics_from_pth(pth_path)
        if entry:
            results.append(entry)
    return results


def print_table(results: list[dict], sort_by: str = 'f1_score', top: int = 0):
    """Print a formatted table of results."""
    if not results:
        print("No results found.")
        return

    # Sort descending (except MAE/ME/RMSE which are lower=better)
    reverse = sort_by not in ('mae', 'me', 'rmse')
    results.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)

    if top > 0:
        results = results[:top]

    # Header
    cols = [
        ('Rank', 4), ('F1', 7), ('F2', 7), ('F5', 7),
        ('Recall', 7), ('Prec', 7), ('MAE', 6), ('ME', 6),
        ('Ep', 3), ('Model', 30), ('Path', 0),
    ]
    header = ''
    for name, width in cols:
        if width:
            header += f'{name:>{width}} '
        else:
            header += name
    print(header)
    print('-' * min(len(header) + 40, 140))

    for rank, r in enumerate(results, 1):
        def fmt(key, width=7, decimals=4):
            val = r.get(key)
            if val is None or val == '?':
                return f'{"?":>{width}}'
            if isinstance(val, float):
                return f'{val:>{width}.{decimals}f}'
            return f'{str(val):>{width}}'

        path = r.get('path', '?')
        # Shorten path for display
        if 'output/' in path:
            path = path[path.index('output/'):]

        model = r.get('model', '?')
        if len(model) > 30:
            model = model[:27] + '...'

        line = (
            f'{rank:>4} '
            f'{fmt("f1_score")} '
            f'{fmt("f2_score")} '
            f'{fmt("f5_score")} '
            f'{fmt("recall")} '
            f'{fmt("precision")} '
            f'{fmt("mae", 6, 2)} '
            f'{fmt("me", 6, 2)} '
            f'{fmt("epoch", 3, 0)} '
            f'{model:<30} '
            f'{path}'
        )
        print(line)


def export_csv(results: list[dict], csv_path: str, sort_by: str = 'f1_score'):
    """Export results to CSV."""
    reverse = sort_by not in ('mae', 'me', 'rmse')
    results.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)

    fieldnames = [
        'rank', 'f1_score', 'f2_score', 'f5_score', 'recall', 'precision',
        'mae', 'me', 'rmse', 'epoch', 'model', 'wandb_run', 'path',
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for rank, r in enumerate(results, 1):
            r['rank'] = rank
            writer.writerow(r)

    print(f"Exported {len(results)} results to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Find and rank best training runs')
    parser.add_argument('--output-dir', '-d', default='./output',
                        help='Output directory to scan for best_model.pth files')
    parser.add_argument('--log', '-l', help='Parse a log file instead of scanning pth files')
    parser.add_argument('--sort-by', '-s', default='f1_score',
                        choices=['f1_score', 'f2_score', 'f5_score', 'recall', 'precision', 'mae', 'me', 'rmse'],
                        help='Metric to sort by (default: f1_score)')
    parser.add_argument('--top', '-t', type=int, default=0,
                        help='Show only top N results (0 = all)')
    parser.add_argument('--csv', '-c', help='Export results to CSV file')

    args = parser.parse_args()

    if args.log:
        results = parse_metrics_from_log(args.log)
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"Error: {output_dir} does not exist")
            sys.exit(1)
        results = scan_output_dir(output_dir)

    print(f"\nFound {len(results)} runs, sorted by {args.sort_by}:\n")
    print_table(results, sort_by=args.sort_by, top=args.top)

    if args.csv:
        print()
        export_csv(results, args.csv, sort_by=args.sort_by)


if __name__ == '__main__':
    main()
