# Wellbore Geology (TVT) Prediction 

---

## Module map
| File | Role | Phase |
|---|---|---|
| `preprocess.py` | Clean 18-feature matrix (short-gap GR interp + masks, geometry, band context) 
| `eda.py` | Per-well summary table + 6-panel report figure 
| `geosteering_starter.py` | Loader + carry / Viterbi baselines 
| `geosteering_cv.py` | **The scoreboard**: GroupKFold + spatial CV, randomized-PS, pooled RMSE 
| `features_and_models.py` | Windowed-correlation feature + GBM **ablation harness** 
| `transformer_train.py` | Dual-encoder cross-attention model + augmentation + inference 
| `ensemble.py` | Blend + physical guardrails 
| `colab_train_cuda.py` | Colab/cuda driver: CV-validate + train final + export weights 
| `kaggle_submission.py` | Self-contained **baseline** submission notebook 


