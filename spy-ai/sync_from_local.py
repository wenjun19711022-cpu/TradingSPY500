"""Mirror the running local system into this repo folder (code, frozen models and research results only).

  C:\\Users\\94868\\spybt\\watch  ->  spy-ai/watch      (live watcher, v6 engine, swing mode, dashboard)
  C:\\Users\\94868\\spybt\\v6     ->  spy-ai/research   (data download, v6 training, trading backtests, strategy factory, swing study)

Never copied: your live database / logs / daily reports (watch/data, watch/reports), market data and caches
(research/data/*.parquet, research/cache), anything over 3 MB.  Run:  py -3 spy-ai/sync_from_local.py"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("C:/Users/94868/spybt/watch", os.path.join(HERE, "watch"), {"data", "reports", "__pycache__"}),
         ("C:/Users/94868/spybt/v6", os.path.join(HERE, "research"), {"cache", "__pycache__", "models"})]
KEEP_EXT = {".py", ".json", ".md", ".bat", ".html", ".pkl", ".csv"}
MAX = 3 * 1024 * 1024
SMALL_DATA = {"events.csv", "okx_spy_funding.csv"}            # research/data files worth versioning


def sync(src, dst, skip_dirs):
    n = 0
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        top = rel.split(os.sep)[0]
        if top in skip_dirs: dirs[:] = []; continue
        if top == "data" and "research" in dst:
            files = [f for f in files if f in SMALL_DATA]
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            p = os.path.join(root, f)
            if os.path.splitext(f)[1].lower() not in KEEP_EXT or os.path.getsize(p) > MAX or f.endswith(".log") or f.startswith("_"): continue
            q = os.path.join(dst, rel, f); os.makedirs(os.path.dirname(q), exist_ok=True)
            shutil.copy2(p, q); n += 1
    return n


if __name__ == "__main__":
    for s, d, skip in PAIRS:
        print("%s -> %s : %d files" % (s, os.path.relpath(d, HERE), sync(s, d, skip)))
