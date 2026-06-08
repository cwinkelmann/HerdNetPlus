"""
Compare AutoML / benchmark results.

Reads JSON result files from benchmark_results/ and prints a ranked table.
Here
Usage:
    python tools/benchmark_compare.py                          # rank all by f1_score
    python tools/benchmark_compare.py --sort-by recall         # sort by recall
    python tools/benchmark_compare.py --baseline dla34_001     # show deltas from baseline
    python tools/benchmark_compare.py --model convnext         # filter by model prefix
    python tools/benchmark_compare.py --results-dir /path/to   # custom results dir
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


METRICS_KEYS = ['f1_score', 'f2_score', 'recall', 'precision', 'mae', 'rmse']


def load_results(results_dir: Path) -> List[dict]:
    """Load all completed trial JSON files from the results directory."""
    results = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        with open(p) as f:
            data = json.load(f)
        if data.get('status') == 'completed':
            results.append(data)
    return results


def print_table(
    results: List[dict],
    sort_by: str = 'f1_score',
    baseline_id: Optional[str] = None,
    model_filter: Optional[str] = None,
):
    """Print a formatted comparison table."""
    if model_filter:
        results = [r for r in results if model_filter.lower() in r.get('trial_id', '').lower()
                   or model_filter.lower() in r.get('config_name', '').lower()]

    if not results:
        print("No completed results found.")
        return

    # Determine sort direction (lower is better for mae/rmse)
    lower_is_better = sort_by in ('mae', 'rmse', 'mse')
    results.sort(
        key=lambda r: r.get('metrics', {}).get(sort_by, float('inf') if lower_is_better else 0),
        reverse=not lower_is_better
    )

    # Find baseline if specified
    baseline = None
    if baseline_id:
        for r in results:
            if baseline_id in r.get('trial_id', ''):
                baseline = r
                break
        if baseline is None:
            print(f"Warning: baseline '{baseline_id}' not found")

    if baseline:
        bm = baseline.get('metrics', {})
        print(f"\nBaseline: {baseline['trial_id']}", end="")
        for k in METRICS_KEYS:
            if k in bm:
                print(f"  {k}={bm[k]:.4f}", end="")
        print("\n")

    # Header
    cols = ['Rank', 'Trial ID', 'Model']
    for k in METRICS_KEYS:
        cols.append(k.upper())
        if baseline:
            cols.append(f'D{k[:4].upper()}')
    cols.append('Time')

    widths = [5, 35, 20] + [8, 7] * len(METRICS_KEYS) if baseline else [5, 35, 20] + [10] * len(METRICS_KEYS)
    widths.append(8)

    # Simplified: just use format strings
    header_parts = [f"{'Rank':<5}", f"{'Trial ID':<35}", f"{'Model':<20}"]
    for k in METRICS_KEYS:
        header_parts.append(f"{k:<10}")
        if baseline:
            header_parts.append(f"{'delta':<8}")
    header_parts.append(f"{'Time':<8}")

    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results):
        m = r.get('metrics', {})
        t = r.get('training_time_seconds', 0)
        time_str = f"{t/60:.0f}m" if t > 60 else f"{t:.0f}s"

        parts = [f"{i+1:<5}", f"{r['trial_id']:<35}", f"{r.get('config_name', ''):<20}"]

        for k in METRICS_KEYS:
            val = m.get(k)
            if val is not None:
                parts.append(f"{val:<10.4f}")
            else:
                parts.append(f"{'N/A':<10}")

            if baseline:
                bval = baseline.get('metrics', {}).get(k)
                if val is not None and bval is not None:
                    delta = val - bval
                    sign = '+' if delta >= 0 else ''
                    parts.append(f"{sign}{delta:<7.4f}")
                else:
                    parts.append(f"{'N/A':<8}")

        parts.append(time_str)
        print(" ".join(parts))

    # Summary
    if len(results) >= 2:
        best = results[0]
        bm = best.get('metrics', {})
        print(f"\nBest: {best['trial_id']} ({sort_by}={bm.get(sort_by, 'N/A')})")


def main():
    parser = argparse.ArgumentParser(description="Compare AutoML benchmark results")
    parser.add_argument('--results-dir', default='benchmark_results',
                        help='Directory containing result JSON files')
    parser.add_argument('--sort-by', default='f1_score',
                        help='Metric to sort by (default: f1_score)')
    parser.add_argument('--baseline', default=None,
                        help='Trial ID (or prefix) to use as baseline for delta comparison')
    parser.add_argument('--model', default=None,
                        help='Filter results by model name or trial prefix')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    results = load_results(results_dir)
    print_table(results, sort_by=args.sort_by, baseline_id=args.baseline, model_filter=args.model)


if __name__ == '__main__':
    main()
