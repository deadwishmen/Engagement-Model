# Engagement Model V5

K-hop Social Graph + Backbone/Time-Attention + Bi-LSTM Skeleton + Multimodal
Self-Attention Fusion engagement classifier.

This repository is a clean, modular refactor of a research notebook
(`engagement-model-final.ipynb`) into an importable Python package. **No modeling
logic was changed** — code was only reorganized into modules, and notebook-level
global variables were turned into explicit function arguments / return values, so
every piece can be tested, reused, or run headlessly, while the whole thing is
still perfectly usable from a notebook (see `notebooks/run_engagement_model.ipynb`).

## Architecture

```
K-hop neighbors --Backbone--> Time Attention --> Fk (multiple nodes) ----\
                                                                           |
Body (16 frames) --Backbone--> Time Attention --> Fb ------------   Build Graph
                                                       |     \      (Fk<->Fk<->Fb)
                                                       |      \          |
                                                       |       \        GNN
                                                       |        \        |
                                                       |         \-------+---> Fb_refined = F_social
Skeleton (16 frames) --Bi-LSTM + Attn--------\         |
                                               Concat(Fb, skeleton) = Ftarget
                                                     |
Face (16 frames) --Backbone--> Time Attention --> Ff |
                                                     | |
                                       F_social, F_context, Ftarget, Ff
                                                     |
                                     Multimodal Self-Attention Fusion
                                                     |
                                                    MLP -> logits
```

Additions beyond the base diagram:
- **FiLM conditioning**: per-frame ResNet features modulate the skeleton branch
  before the Bi-LSTM, injecting visual context into otherwise purely-geometric pose
  features.
- **Context Window**: a temporal analogue of K-hop — clips before/after the current
  one, for the *same* person — fused the same way (GNN + gated residual).
- **Auxiliary heads**: optional multi-task supervision on pose/action/object/
  interaction/emotion labels (gradient-detached from the main encoders).
- **Ordinal auxiliary loss** (CORAL-style) and **Supervised Contrastive loss** on the
  fused embedding, both used only during training.
- **Track-level mode**: an optional Bi-LSTM over an entire track's segments
  (`TrackEngagementModel`), as an alternative to independent per-segment scoring.

## Repository layout

```
engagement_model/
├── pyproject.toml, requirements.txt      # packaging / dependencies
├── configs/default_paths.json            # example path overrides
├── scripts/
│   ├── train.py                          # CLI: run the full pipeline
│   └── evaluate.py                       # CLI: evaluate an existing checkpoint
├── notebooks/
│   └── run_engagement_model.ipynb        # thin, runnable notebook wrapper
└── src/engagement_model/
    ├── config.py                         # CONFIG dict, seeding, device setup
    ├── losses.py                         # Focal loss, ordinal/SupCon/aux losses
    ├── pipeline.py                       # top-level orchestrator
    ├── data/
    │   ├── manifest.py                   # manifest loading + label maps
    │   ├── graph.py                      # K-hop neighbor graph + context window
    │   ├── diagnostics.py                # leakage / label-stability / cache checks
    │   ├── filtering.py                  # filter by cached-feature availability
    │   └── dataset.py                    # Dataset classes + DataLoader builder
    ├── models/
    │   ├── temporal_encoder.py           # Backbone + Time-Attention encoder
    │   ├── skeleton_encoder.py           # Bi-LSTM + attention-pooling skeleton encoder
    │   └── engagement_model.py           # GNN, FiLM, fusion transformer, full model
    ├── training/
    │   ├── setup.py                      # model/optimizer/scheduler/AMP builders
    │   ├── engine.py                     # train/eval steps, bias tuning, diagnostics
    │   └── loop.py                       # epoch loop with early stopping
    └── evaluation/
        ├── social_diagnostics.py         # does the model actually use neighbors?
        └── test_eval.py                  # final test-set evaluation + plots
```

## Installation

```bash
cd engagement_model
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
pip install -e .                                     # optional, for `import engagement_model` anywhere
```

## Usage

### Option A — command line

```bash
# Edit configs/default_paths.json with your OUTPUT_DIR / FEATURE_DIR / SKELETON_DIR /
# SPLIT_NAMES_JSON_PATH / CHECKPOINT_DIR, then:
python scripts/train.py --config configs/default_paths.json

# Or override individual paths/hyperparameters directly:
python scripts/train.py \
    --output-dir /data/engagement-dataset-npy \
    --feature-dir /data/resnet_features \
    --skeleton-dir /data/skeleton_features \
    --checkpoint-dir /data/checkpoints_v5 \
    --epochs 25 --batch-size 16

# Quick smoke test on a tiny subset (2 epochs):
python scripts/train.py --config configs/default_paths.json --dry-run

# Evaluate an existing checkpoint without retraining:
python scripts/evaluate.py --config configs/default_paths.json \
    --checkpoint /data/checkpoints_v5/best_model_v4_social_interaction.pt
```

Any key from `engagement_model.config.DEFAULT_CONFIG` can be overridden either via
the JSON config file or with a repeatable `--set KEY=VALUE` flag, e.g.
`--set LR=5e-5 --set USE_SUPCON_LOSS=false`.

### Option B — notebook / interactive Python

```python
from engagement_model.pipeline import run_full_pipeline

result = run_full_pipeline(config_overrides={
    "OUTPUT_DIR": "...", "FEATURE_DIR": "...", "SKELETON_DIR": "...",
    "SPLIT_NAMES_JSON_PATH": "...", "CHECKPOINT_DIR": "...",
})
print(result["train_result"]["best_val_macro_f1"])
print(result["test_result"]["test_metrics"]["macro_f1"])
```

See `notebooks/run_engagement_model.ipynb` for both a one-call version and a
step-by-step version (useful for inspecting the manifest dataframe, the neighbor
graph, or a single batch before committing to a full training run) — it runs in
Jupyter, JupyterLab, Kaggle, Colab, or any other notebook environment.

### Option C — import individual pieces

Every module is independently importable, e.g.:

```python
from engagement_model.data.manifest import load_manifest_and_prepare
from engagement_model.data.graph import build_neighbor_index
from engagement_model.models.engagement_model import EngagementModelV5
```

## Data expectations

The pipeline expects, under `CONFIG["OUTPUT_DIR"]`:
- `manifest/manifest.{csv,parquet,jsonl}` — one row per 16-frame clip/segment, with
  `sample_id`, `engagement_label`, `session`, `camera_id`, `object_id`, `bbox_norm`/
  `bbox_pixel`, `clip_path`, and optional `pose_label`/`act_label`/`obj_label`/
  `int_label`/`emo_label` auxiliary columns.
- `CONFIG["FEATURE_DIR"]/features_body/<sample_id>.npy` and
  `CONFIG["FEATURE_DIR"]/features_face/<sample_id>.npy` — cached ResNet-50 features,
  shape `(NUM_FRAMES, FEATURE_DIM)`.
- `CONFIG["SKELETON_DIR"]/<sample_id>.npy` — cached YOLO-Pose/COCO-17 skeleton
  (`keypoints`, `keypoint_scores`, `bbox`, `detected`).
- (optional) `CONFIG["SPLIT_NAMES_JSON_PATH"]` — a `{"train": [...], "val": [...],
  "test": [...]}` JSON of session filenames, to override the manifest's own `split`
  column with a session-level split (avoids train/val/test leakage of the same
  track).

## Notes on fidelity

This refactor is a straight reorganization: every function/class body is
byte-for-byte the same computation as the original notebook cells; only
notebook-global state (`CONFIG`, `df`, `DEVICE`, `AUX_LABEL2ID`, etc.) was replaced
with explicit parameters and dict-based return values so the code has no hidden
order-of-cell-execution dependencies. All modules have been import- and
forward-pass-tested with synthetic tensors matching the real feature shapes.
