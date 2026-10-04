"""Build the single-file dashboard from results/study.json.

    python -m spyopt.report.dashboard                       # -> dashboard/index.html
    python -m spyopt.report.dashboard --fragment out.html   # body-only variant (for hosts that
                                                          #    wrap the page in their own <html>)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("template.html")


def render(study: dict) -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(study, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return html.replace("/*__STUDY__*/null", payload)


def full_document(fragment: str) -> str:
    return ("<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + fragment + "\n</body>\n</html>\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=str(ROOT / "results" / "study.json"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "index.html"))
    ap.add_argument("--fragment", help="also write the body-only variant here")
    a = ap.parse_args(argv)
    study = json.loads(Path(a.study).read_text(encoding="utf-8"))
    frag = render(study)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full_document(frag), encoding="utf-8")
    print("wrote", out, f"{out.stat().st_size / 1e6:.2f} MB")
    if a.fragment:
        Path(a.fragment).write_text(frag, encoding="utf-8")
        print("wrote", a.fragment)


if __name__ == "__main__":
    main()
