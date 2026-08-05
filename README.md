# Wellbore Geology (TVT) Prediction — Master Runbook (Phases A–E)

One competition submission + one CS7643 project, from one codebase. 

---

## The one idea everything rests on
TVT is the drill bit's **stratigraphic position** (where in the layer-cake it sits), not its depth. The well is *steered* to stay in a target zone, so TVT is **band-limited and mean-reverting**. Consequences, measured on real wells:
- Following depth (Z) → RMSE 54–97 ft (useless). Geometry alone fails.
- Carry-forward last known TVT → **7.5–15 ft**. This is the bar to beat.
- The signal is **matching the horizontal GR wiggle to the typewell's GR-vs-TVT curve** — differentiable log-correlation. Cross-attention *is* that match.

---

## Module map
| File | Role | Phase |
|---|---|---|
| `preprocess.py` | Clean, test-legal 18-feature matrix (short-gap GR interp + masks, geometry, band context) 
| `eda.py` | Per-well summary table + 6-panel report figure 
| `geosteering_starter.py` | Loader + carry / Viterbi baselines 
| `geosteering_cv.py` | **The scoreboard**: GroupKFold + spatial CV, randomized-PS, pooled RMSE 
| `features_and_models.py` | Windowed-correlation feature + GBM **ablation harness** 
| `transformer_train.py` | Dual-encoder cross-attention model + augmentation + inference 
| `ensemble.py` | Blend + physical guardrails 
| `colab_train_cuda.py` | Colab driver: CV-validate + train final + export weights 
| `kaggle_submission.py` | Self-contained **baseline** submission notebook | A |


