# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.3",
#     "transformers>=5",
#     "flash-linear-attention",
#     "huggingface_hub>=0.24",
#     "numpy>=1.24",
#     "pyarrow>=15",
# ]
# ///
"""Score a decider model (Mapika/decider-4b or a fine-tune of it) on the test split laya_pipeline built, with the same
metrics and groups as laya_pipeline's eval_report.json, so the two model families compare row for row.

  uv run --script scripts/decider_eval.py --data laya_run/data/all/test-00000-of-00001.parquet --out decider_eval.json
      [--model Mapika/decider-4b] [--revision <sha>] [--limit N]

The questions go in as they are: decider's system_one takes the same choice / noul / score schema as Laya.
"""
import argparse, json, math, random, sys, time
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def ece_score(conf, correct, bins=15):
    """laya.common.ece_score, so the numbers compare with eval_report.json."""
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def summarize(rows):
    conf, correct = np.array([r["conf"] for r in rows]), np.array([r["correct"] for r in rows])
    return {"n": len(rows), "accuracy": round(float(correct.mean()), 4), "brier": round(float(np.mean([r["brier"] for r in rows])), 4),
            "nll": round(float(np.mean([r["nll"] for r in rows])), 4), "ece": round(ece_score(conf, correct), 4)}


def auroc(p, y):
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def probs_in_gold_order(answer, gold):
    """decider's answer as probabilities over the gold's keys (choice keys, "0".."n-1", "false"/"true")."""
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    p = {str(k): v for k, v in answer["probabilities"].items()}
    return [p[str(k)] for k in gold["probabilities"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Mapika/decider-4b")
    ap.add_argument("--revision", default="eb5fbdfc9448473ec25e399882912863afbdb70e")   # v2.1, as measured
    ap.add_argument("--limit", type=int, default=0, help="score a random sample of N rows (0: all)")
    ap.add_argument("--rows_per_call", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=16384, help="padded tokens per forward pass")
    ap.add_argument("--fp8", action="store_true", help="e4m3 weights and per-token activation scales on the big linears "
                    "(decider.fp8, as DECIDER_FP8=1 serves): measures what FP8 costs in accuracy")
    ap.add_argument("--graphs", action="store_true", help="decider's CUDA-graph engine: fast for one shape, slow to capture many")
    a = ap.parse_args()

    path = a.model if a.model.startswith((".", "/")) else snapshot_download(a.model, revision=a.revision)
    sys.path.insert(0, path)                                   # decider/ ships inside the model repo
    from decider.infer import Decider
    from decider import temperature as TT
    from decider.systemone import assemble
    from decider.model import collate
    import torch

    d = Decider(path, use_graphs=a.graphs)
    fp8_layers = 0
    if a.fp8:
        from decider.fp8 import convert_to_fp8
        fp8_layers = convert_to_fp8(d.eng.core if d.eng is not None else d.m.lm.model)
        print(f"FP8: {fp8_layers} linear layers", flush=True)

    @torch.no_grad()
    def score(its, T):
        """One probability list per slot of each item (as Decider._system_one_probs, without its per-request batching)."""
        if d.eng is not None:
            return d.eng.score_items(its, temperature=T)
        bt = collate(its, d.m.tok.pad_token_id)
        lg = d.m.slot_logits(*[bt[k].to(d.dev) for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")])
        pr, out, c = TT.scaled_softmax(lg, TT.slot_temperatures(T, its)).float().cpu(), [], 0
        for it in its:
            out.append(pr[c:c + len(it["slots"])]); c += len(it["slots"])
        return out
    rows = pq.read_table(a.data).to_pylist()
    if a.limit:
        rows = random.Random(0).sample(rows, min(a.limit, len(rows)))
    print(f"{d.name} on {len(rows):,} rows ({sum(int(r['n_questions']) for r in rows):,} decisions)", flush=True)

    groups, skipped, yes_no = defaultdict(list), Counter(), defaultdict(lambda: ([], []))   # per source: P(yes), gold
    t0 = time.time()
    for c in range(0, len(rows), a.rows_per_call):
        chunk, plans, items = rows[c:c + a.rows_per_call], [], []
        for r in chunk:
            state, questions = json.loads(r["state"]), json.loads(r["questions"])
            try:
                rqs, index, its = d._system_one_items(state, questions)
            except ValueError as e:                            # e.g. a score question with more than 10 levels
                skipped[str(e)[:60]] += len(questions); continue
            plans.append((r, rqs, index, len(items), len(its))); items += its
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))     # similar lengths share a forward pass
        T = TT.for_items(d.T, d.T_by_type, [items[i] for i in order])
        flat, i = [None] * len(items), 0
        while i < len(order):
            per = max(1, a.max_tokens // len(items[order[min(len(order) - 1, i + 63)]]["ids"]))
            j = min(len(order), i + min(per, 64))
            out = score([items[k] for k in order[i:j]], TT.item_slice(T, i, j))
            for k, ps in zip(order[i:j], out):
                flat[k] = [p.tolist() for p in ps]
            i = j
        for r, rqs, index, s, n in plans:
            answers = assemble(rqs, index, [p for ps in flat[s:s + n] for p in ps])
            gold = json.loads(r["gold"])
            for qid, g in gold.items():
                p = np.clip(np.array(probs_in_gold_order(answers[qid], g)), 1e-12, 1); p = p / p.sum()
                t = np.array(list(g["probabilities"].values()), dtype=float)
                row = {"correct": float(p.argmax() == t.argmax()), "conf": float(p.max()),
                       "brier": float(((p - t) ** 2).sum()), "nll": float(-(t * np.log(p)).sum())}
                qtype, workflow = g["type"], r["workflow"]
                source = workflow.split("/")[0]
                for k in ("all", f"type/{qtype}", f"source/{source}", f"source/{source}/{qtype}", f"workflow/{workflow}"):
                    groups[k].append(row)
                if qtype == "noul":
                    for k in (source, workflow):
                        yes_no[k][0].append(p[1]); yes_no[k][1].append(t.argmax())
        done = min(c + a.rows_per_call, len(rows))
        print(f"{done:,} / {len(rows):,} rows  {time.time() - t0:.0f}s  accuracy so far {np.mean([x['correct'] for x in groups['all']]):.3f}", flush=True)

    report = {g: summarize(v) for g, v in sorted(groups.items())}
    for k, (p, y) in sorted(yes_no.items()):              # noul answers as answerability: both error kinds apart
        p, y = np.array(p), np.array(y)
        entry = {"n": len(p), "acc_answerable": round(float((p[y == 1] >= .5).mean()), 4) if (y == 1).any() else None,
                 "acc_unanswerable": round(float((p[y == 0] < .5).mean()), 4) if (y == 0).any() else None}
        if (y == 1).any() and (y == 0).any():
            entry["auroc"] = round(auroc(p, y), 4)
        report[f"{k}/answerability"] = entry
    json.dump({"model": d.name, "path": a.model, "revision": a.revision, "fp8_layers": fp8_layers, "limit": a.limit,
               "skipped": dict(skipped),
               "seconds": round(time.time() - t0), "report": report}, open(a.out, "w"), indent=1)
    for k in ["all"] + [k for k in report if k.startswith("type/") or (k.startswith("source/") and k.count("/") == 1)]:
        print(k, report[k], flush=True)
    for k in [k for k in report if k.endswith("/answerability") and k.count("/") == 1]:
        print(k, report[k], flush=True)
    print("skipped decisions:", dict(skipped))


if __name__ == "__main__":
    main()
