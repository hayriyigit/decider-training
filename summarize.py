#!/usr/bin/env python3
"""A short report of a round, to paste back instead of the logs: training status, the starting vs the trained model on the
round's test split (the categories that matter), TTL per class, and system-one-bench-tr accuracy by area next to the
bench's published models. Standard library only.

  python3 summarize.py configs/decider_judge4.yaml
"""
import json, re, sys
from pathlib import Path

CONFIG = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/decider_judge4.yaml")
NAME = CONFIG.stem
WORK = Path(re.search(r"^workdir:\s*(\S+)", CONFIG.read_text(), re.M).group(1))
START, NEW = Path(f"{NAME}_eval_start.json"), WORK / "eval_report.json"

# the rows that tell whether the round did what it should and broke nothing
ROWS = [
    ("ttl_tr/realtime", "TTL realtime"), ("ttl_tr/hourly", "TTL hourly"), ("ttl_tr/daily", "TTL daily"), ("ttl_tr/stable", "TTL stable"),
    ("nomiracl_tr/tam", "NoMIRACL relevant (yes)"), ("nomiracl_tr/katkisiz", "NoMIRACL not relevant (no)"),
    ("squad_tr/tam", "SQuAD-tr answerable (yes)"), ("squad_tr/metrik", "SQuAD-tr unanswerable (no)"),
    ("boolq_tr/evet_hayir", "BoolQ-tr evidence (yes)"),
    ("synth2_tr/kismi_evet_hayir", "synth v2 yes/no, partial (yes)"), ("synth2_tr/varlik", "synth v2 other entity (no)"),
    ("synth2_tr/kurum", "synth v2 other institution (no)"), ("synth2_tr/donem", "synth v2 wrong period (no)"),
    ("hotpot_tr/kismi", "HotpotQA partial (yes)"), ("hotpot_tr/katkisiz", "HotpotQA no contribution (no)"),
    ("template_tr/headlines", "headline-list page (no)"),
]
# system-one-bench-tr README, test split: accuracy on the 25 main tasks and by area
AREAS = {
    "routing": ["massive_intent_tr", "xsid_tr", "news_topic_tr"],
    "opinion": ["buyuksinema_tr", "musteri_yorumlari_tr", "biltweetnews_tr", "tweet_emotion_tr"],
    "safety": ["offenseval_tr", "sms_spam_tr", "hateday_tr"],
    "fact_checking": ["xfact_teyit_tr", "checkthat_tr", "trclaim19_tr"],
    "judging": ["seahorse_tr", "webfaq_tr"],
    "workflow": ["jev_noul_tr", "jev_noul_unseen_tr"],
    "politics": ["touche_tr"],
    "language": ["nli_tr", "stsb_tr", "belebele_tr", "xcopa_tr", "turblimp_tr"],
    "knowledge": ["include_tr", "global_mmlu_tr", "turkishmmlu_tr"],
}
MAIN = [t for ts in AREAS.values() for t in ts if t != "massive_intent_tr"]
REFERENCE = {   # main, then the areas in AREAS' order
    "Jev": [77.6, 87.4, 62.4, 90.6, 51.2, 81.7, 87.3, 86.8, 78.9, 86.9],
    "vistalab-system-one-e4b": [75.4, 87.8, 63.3, 92.1, 57.5, 79.5, 89.7, 76.6, 76.3, 66.4],
    "Metask Jev 4B": [69.7, None, 53.6, 91.4, 55.1, 74.8, 78.9, 74.4, 68.1, 62.1],
    "Qwen3.5 4B": [65.9, 75.9, 49.9, 67.7, 52.1, 78.9, 82.9, 64.1, 67.9, 63.8],
}


def f(x):
    return "  -  " if x is None else f"{x:.3f}"


def training():
    print(f"## {NAME}\n")
    log = Path(f"logs/{NAME}_lora.log")
    if not log.exists():
        print("no training log yet"); return
    text = log.read_text()
    steps = re.findall(r"step=(\d+) of=(\d+) ce=([\d.]+) kl=([\d.]+) acc=([\d.]+) .*tok_per_s=(\d+) eta_h=([\d.]+)", text)
    if steps:
        s = steps[-1]
        print(f"- last step {s[0]}/{s[1]}: ce {s[2]}, kl {s[3]}, train acc {s[4]}, {s[5]} tok/s, eta {s[6]} h")
    for pat, label in [(r"validation .*val_slot_accuracy=([\d.]+) val_ce=([\d.]+)", "validation (acc, ce)"),
                       (r"temperatures on .*?: (\{.*?\})", "temperatures"), (r"^EXIT (\d+)", "exit code")]:
        m = re.findall(pat, text, re.M)
        if m:
            print(f"- {label}: {m[-1]}")
    for line in text.splitlines():
        if re.search(r"Traceback|Error|out of memory", line):
            print(f"- ERROR: {line.strip()[:200]}")
    print()


def evaluation():
    if not START.exists() or not NEW.exists():
        print(f"evaluation: {'start ok' if START.exists() else 'no start eval'}, {'new ok' if NEW.exists() else 'no final eval'}\n"); return
    a, b = json.loads(START.read_text())["report"], json.loads(NEW.read_text())
    b = b.get("report", b)
    print("| test rows | n | start | new | change |\n|---|---:|---:|---:|---:|")
    for key, label in [("all", "all")] + [(f"source/{s}", s) for s in sorted({k.split('/')[1] for k in b if k.startswith('source/') and k.count('/') == 1})]:
        if key in a and key in b:
            print(f"| {label} | {b[key]['n']} | {f(a[key]['accuracy'])} | {f(b[key]['accuracy'])} | {b[key]['accuracy'] - a[key]['accuracy']:+.3f} |")
    print("\n| key category | n | start | new | change |\n|---|---:|---:|---:|---:|")
    for wf, label in ROWS + [(k[len("workflow/"):], k[len("workflow/"):]) for k in sorted(b) if k.startswith("workflow/ragbench_tr/")]:
        k = f"workflow/{wf}"
        if k in a and k in b:
            d = b[k]["accuracy"] - a[k]["accuracy"]
            print(f"| {label} | {b[k]['n']} | {f(a[k]['accuracy'])} | {f(b[k]['accuracy'])} | {d:+.3f}{'  <-- dropped' if d <= -0.03 else ''} |")
    print(f"\nECE all: {f(a['all']['ece'])} -> {f(b['all']['ece'])}\n")


def bench():
    root = Path("bench")
    models = sorted(p.name for p in (root / "results").iterdir() if p.is_dir()) if (root / "results").is_dir() else []
    if not models:
        print("bench: no results yet"); return

    def acc(model, task):
        out, data = root / "results" / model / task / "test.jsonl", root / "data" / task / "test.jsonl"
        if not out.exists() or not data.exists():
            return None
        gold = {r["id"]: r["gold"] for r in map(json.loads, open(data, encoding="utf-8"))}
        ok = n = 0
        for r in map(json.loads, open(out, encoding="utf-8")):
            for q, p in r["probs"].items():
                if p is None:
                    continue
                g = gold[r["id"]][q]
                g = str(g).lower() if isinstance(g, bool) else str(g)
                ok += max(p, key=p.get) == g; n += 1
        return 100 * ok / n if n else None

    def mean(xs):
        return None if not xs or any(x is None for x in xs) else sum(xs) / len(xs)

    print("system-one-bench-tr, test accuracy (%): main = 25 tasks; the bench's published models for reference\n")
    print("| model | main | " + " | ".join(AREAS) + " |\n|---|" + "---:|" * (len(AREAS) + 1))
    for m in models:
        per = {t: acc(m, t) for ts in AREAS.values() for t in ts}
        cells = [mean([per[t] for t in MAIN])] + [mean([per[t] for t in ts]) for ts in AREAS.values()]
        print(f"| **{m}** | " + " | ".join("-" if c is None else f"{c:.1f}" for c in cells) + " |")
    for m, cells in REFERENCE.items():
        print(f"| {m} | " + " | ".join("-" if c is None else f"{c:.1f}" for c in cells) + " |")
    missing = [t for t in MAIN if acc(models[-1], t) is None]
    if missing:
        print(f"\nnot answered yet by {models[-1]}: {', '.join(missing)}")


training()
evaluation()
bench()
