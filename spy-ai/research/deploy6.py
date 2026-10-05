"""Copy the v6 engine + frozen models + backtest results into the watcher (watch/model6/)."""
import os, shutil

HERE = os.path.dirname(os.path.abspath(__file__)); DST = "C:/Users/94868/spybt/watch/model6"
os.makedirs(DST, exist_ok=True)
files = ["ind.py", "feat6.py", "models6.py", "models/v6_bottom.pkl", "models/v6_top.pkl", "models/v6_features.json",
         "results/train6.json", "results/bt6.json", "results/plan.json", "results/v7.json", "results/momentum.json", "results/factory_summary.json", "results/swing_final.json", "results/zigzag_review.json"]
for f in files:
    src = os.path.join(HERE, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DST, os.path.basename(f))); print("copied", f)
    else:
        print("MISSING", f)
