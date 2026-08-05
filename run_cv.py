from geosteering_cv import run_cv, CarryPredictor, ViterbiPredictor
ROOT = r"train"
run_cv(ROOT, [CarryPredictor, ViterbiPredictor], k=5, mode="real", fold_kind="group")