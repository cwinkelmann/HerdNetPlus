# False Positive Correction Workflow

## Problem

The ConvNeXt model (F1=0.95) detects iguanas that are missing from the ground truth annotations. These high-confidence "false positives" are actually real iguanas — the dataset needs correction, not the model.

## Workflow Overview

```
error_analysis.ipynb          Label Studio              Hasty.ai JSON
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Run inference    │     │ Human reviews    │     │ Single source    │
│ Identify FPs     │────▶│ each FP image    │────▶│ of truth updated │
│ Upload to LS     │     │ Confirm / reject │     │ CSV regenerated  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

## Steps

### 1. Detect False Positives

Run `notebooks/error_analysis.ipynb` with the best model checkpoint. The notebook:
- Loads cached inference outputs (no GPU needed for re-runs)
- Matches predictions to GT using nearest-neighbor at configurable radius
- Identifies FPs sorted by confidence score
- Exports FP locations as CSV with full-image coordinates

### 2. Upload FPs to Label Studio

The notebook uploads FP crop images to a running Label Studio instance with:
- **Red points**: model predictions (the FPs to review)
- **Green points**: existing GT annotations (for context)

The reviewer's task is simple:
- See an iguana at the red point → **submit** (confirms the detection)
- No iguana there → **delete the point** and submit

### 3. Download Corrections

Export confirmed points from Label Studio. The notebook converts them to Hasty.ai format with:
- `class_name: "iguana_point"`
- `attributes: {"cvat": "new"}` — marks as added during correction
- `created_by` / `update_date` — tracks who corrected and when
- Preserves the label UUID system from the active-learning repo

### 4. Merge into Hasty JSON

Confirmed FPs are appended to `hasty_format.json`:
- Each new point gets a UUID label ID
- Inserted into the correct image entry (matched by `image_name`)
- The Hasty JSON remains the single source of truth

### 5. Regenerate HerdNet CSV

The updated `hasty_format.json` is converted to `herdnet_format.csv`:
```
images,x,y,species,labels
FMO05___DJI_0317_x3406_y1048.jpg,416,496,iguana,1   ← newly added
```

## Dependencies

- **HerdNet** (`animaloc`) — model inference + cached outputs
- **active-learning** (`pip install -e /home/christian/hnee/active-learning`) — Hasty JSON types (`HastyAnnotationV2`, `ImageLabel`, `Keypoint`) + keypoint ID mappings
- **Label Studio** — running instance at `http://localhost:8080`
- Label Studio upload/download implemented directly in the notebook (simple REST API calls, no external package needed)

### Key imports from active-learning

```python
from com.biospheredata.types.HastyAnnotationV2 import (
    HastyAnnotationV2, AnnotatedImage, ImageLabel, Keypoint, LabelClass
)
from active_learning.config.mapping import keypoint_id_mapping
```

## File Locations

| File | Purpose |
|------|---------|
| `notebooks/error_analysis.ipynb` | Main workflow notebook |
| `output/hparam_search/cache/*.pt` | Cached model outputs |
| `output/hparam_search/detection_errors.csv` | Exported FP/FN locations |
| `data_FMO03_02_05/val/crops_512/hasty_format.json` | GT annotations (source of truth) |
| `data_FMO03_02_05/val/crops_512/herdnet_format.csv` | Training CSV (regenerated) |

## Future Work

- Apply same workflow to training set (larger, more missing annotations likely)
- Apply to other island datasets (Floreana, Fernandina, Genovesa)
- Automate as a post-training step: train → detect FPs → correct → retrain