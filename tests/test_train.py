"""
Training pipeline tests.

Uses HuggingFace sample data (downloaded + patched in conftest.py).
Runs 2-epoch training with each model architecture to verify the full pipeline.
"""
import torch
from pathlib import Path

import pytest
from animaloc.utils.train import main


def _detect_device():
    """Pick the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _common_overrides(training_data, tmp_output_dir):
    """Return overrides shared by all training tests."""
    return [
        f"datasets.train.csv_file={training_data['train_csv']}",
        f"datasets.train.root_dir={training_data['train_root']}",
        f"datasets.validate.csv_file={training_data['val_csv']}",
        f"datasets.validate.root_dir={training_data['val_root']}",
        "training_settings.epochs=2",
        "training_settings.batch_size=2",
        "training_settings.num_workers=0",
        "training_settings.warmup_iters=1",
        "wandb_flag=False",
        "model.load_from=null",
        f"hydra.run.dir={tmp_output_dir}",
        f"device_name={_detect_device()}",
    ]


@pytest.fixture
def tmp_output_dir(tmp_path):
    return str(tmp_path / "output")


def test_train_dla34(load_config, training_data, tmp_output_dir):
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
    ]
    cfg = load_config("dla34_delplanque", overrides=overrides)
    results, metrics = main(cfg)
    assert results is not None
    assert 'f1_score' in metrics


def test_train_timm_dla34(load_config, training_data, tmp_output_dir):
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
    ]
    cfg = load_config("dla34_timm", overrides=overrides)
    results, metrics = main(cfg)
    assert results is not None
    assert 'f1_score' in metrics


@pytest.mark.slow
def test_train_convnext_camouflaged(load_config, training_data, tmp_output_dir):
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
    ]
    cfg = load_config("convnext_camouflaged", overrides=overrides)
    results, metrics = main(cfg)
    assert results is not None
    assert 'f1_score' in metrics


def _read_training_log(work_dir):
    """Return concatenated contents of all *_training.log files in work_dir."""
    logs = list(Path(work_dir).glob("*_training.log"))
    assert logs, f"No *_training.log found in {work_dir}"
    return "\n".join(p.read_text() for p in logs)


@pytest.mark.slow
def test_early_stopping_default_patience_does_not_trigger(
    load_config, training_data, tmp_output_dir, monkeypatch
):
    """Enabling early_stopping with no explicit patience must fall back to a
    sane integer default (10), not the boolean `False`. With only 2 epochs,
    training must run to completion — no early-stop log line should appear.

    Pre-fix behavior: patience defaults to `False` (== 0), so `wait >= patience`
    is True from the first validation and training stops at epoch 1.
    """
    Path(tmp_output_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_output_dir)
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
        "training_settings.epochs=2",
        "++training_settings.early_stopping=True",
        # Deliberately omit patience — exercises the fallback path.
    ]
    cfg = load_config("dla34_delplanque", overrides=overrides)
    work_dir, _ = main(cfg)
    log_text = _read_training_log(work_dir)
    assert "Early stopping triggered at epoch" not in log_text, (
        "Default fallback should prevent early stopping from firing in 2 epochs; "
        "if this fires, the patience fallback is wrong."
    )


@pytest.mark.slow
def test_early_stopping_triggers_when_metric_does_not_improve(
    load_config, training_data, tmp_output_dir, monkeypatch
):
    """With patience=1 and an unreachable min_delta (F1 is in [0,1]),
    no validation can register as an improvement, so training must stop
    at the first validation epoch.
    """
    Path(tmp_output_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_output_dir)
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
        "training_settings.epochs=3",
        "++training_settings.early_stopping=True",
        "++training_settings.early_stopping_patience=1",
        "++training_settings.early_stopping_min_delta=10.0",
    ]
    cfg = load_config("dla34_delplanque", overrides=overrides)
    work_dir, _ = main(cfg)
    log_text = _read_training_log(work_dir)
    assert "Early stopping triggered at epoch" in log_text, (
        "Patience=1 with unreachable min_delta should stop at epoch 1."
    )
