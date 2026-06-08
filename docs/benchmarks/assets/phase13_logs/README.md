# Phase-13 training logs (archive)

Loguru file sinks captured from each Phase-13 training run, gzipped for storage. See [`docs/benchmarks/phase13_data_scaling.md`](../../phase13_data_scaling.md) for the analysis these back.

## Files

| File | N | Run date | Source path | Compressed | Raw |
|---|---|---|---|---|---|
| `phase13_N19_s42_training.log.gz`   |   19 | 2026-05-08 | `output/phase13_N19_s42/2026-05-08/14-23-19/`   |  37 KB |  77 KB |
| `phase13_N38_s42_training.log.gz`   |   38 | 2026-05-08 | `output/phase13_N38_s42/2026-05-08/15-54-55/`   |  63 KB | 121 KB |
| `phase13_N76_s42_training.log.gz`   |   76 | 2026-05-08 | `output/phase13_N76_s42/2026-05-08/18-21-11/`   | 109 KB | 213 KB |
| `phase13_N152_s42_training.log.gz`  |  152 | 2026-05-08 | `output/phase13_N152_s42/2026-05-08/22-38-33/`  | 189 KB | 365 KB |
| `phase13_N304_s42_training.log.gz`  |  304 | 2026-05-09 | `output/phase13_N304_s42/2026-05-09/06-39-09/`  | 343 KB | 641 KB |
| `phase13_N608_s42_training.log.gz`  |  608 | 2026-05-09 | `output/phase13_N608_s42/2026-05-09/22-09-34/`  | 651 KB | 1.3 MB |
| `phase13_N1216_s42_training.log.gz` | 1216 | 2026-05-11 | `output/phase13_N1216_s42/2026-05-11/04-45-05/` | 1.3 MB | 2.4 MB |
| `phase13_N2432_s42_training.log.gz` | 2432 | 2026-05-13 | `output/phase13_N2432_s42/2026-05-13/18-05-47/` | 1.3 MB | 2.4 MB |
| `phase13_Nfull_s42_training.log.gz` | full | 2026-05-16 | `output/phase13_Nfull_s42/2026-05-16/05-50-52/` | 1.5 MB | 2.7 MB |

The N=full Docker (remote) run lives only in wandb — see [`hn_phase13_data_scaling`](https://wandb.ai/karisu/hn_phase13_data_scaling) run `20260513_phase13_Nfull_s42_docker_j855ekjq` for the streaming log archive.

## Viewing

```bash
# Inspect a log without unpacking
zcat phase13_N304_s42_training.log.gz | less

# Grep across all of them
zgrep -h "best F1" phase13_*_training.log.gz | sort -u

# Tail the asymptote run
zcat phase13_Nfull_s42_training.log.gz | tail -200
```

## Notes on the "failed" runs

The two runs whose wandb `state=failed` (`N=2432`, local `N=full`) **both crashed on the same numerical-stability issue**: cross-entropy auxiliary loss going NaN mid-epoch, triggering the trainer's `Loss is nan, stopping training` abort.

- `N=2432`: NaN at epoch 15, step 38341/45600. `best_model.pth` from `best_epoch=10` is intact and reported in wandb summary.
- `N=full` (local): NaN at epoch 6, step 103841/129525. `best_model.pth` from `best_epoch=4` is intact.

The remote Docker `N=full` run on identical configuration **completed cleanly** (best_epoch=12, F1=0.882) — the NaN appears to be sporadic at very large training set sizes, not a config bug. A `grad_clip` or `ce_loss` lower bound is a reasonable mitigation if this recurs.
