# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.3",
#     "transformers>=5",
#     "flash-linear-attention",
#     "huggingface_hub>=0.24",
#     "numpy>=1.24",
# ]
# ///
"""A decider model on system-one-bench-tr (github.com/UlkuTuncerKucuktas/system-one-bench-tr): reads the bench's built tasks
(<bench>/data/<task>/{dev,test}.jsonl), answers every question the way the bench asks Jev (one request per question, the
item's state and that question, decider's own prompt), and writes <bench>/results/<name>/<task>/<split>.jsonl in the format
`s1bench run` writes, so `s1bench score` scores it next to the other models.

  uv run --script scripts/s1bench_decider.py --bench bench --model hayriyigit/decider-4b-tr-judge-ttl --revision 8b69f8a [--tasks a b]
A question decider cannot take (e.g. a score question with more levels than it serves) is written as null, as the bench
stores a model's skipped questions; a task already written is skipped, so a rerun continues.
"""
import argparse, json, sys, time
from pathlib import Path

from huggingface_hub import snapshot_download


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.rename(path)                                           # a crash never leaves a half-written split behind


def as_probs(answer):
    """decider's answer in the bench's keys: "true"/"false", the choice keys, the score level indices."""
    if answer["type"] == "noul":
        return {"true": float(answer["noul"]), "false": 1 - float(answer["noul"])}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="the bench folder, holding data/ (s1bench build)")
    ap.add_argument("--model", required=True, help="a Hub repo or a local folder")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", default=None, help="results/<name>; default: the model's last path part")
    ap.add_argument("--tasks", nargs="*", default=None, help="default: every task in data/")
    ap.add_argument("--max_tokens", type=int, default=16384, help="padded tokens per forward pass")
    a = ap.parse_args()

    path = a.model if Path(a.model).is_dir() else snapshot_download(a.model, revision=a.revision)
    sys.path.insert(0, path)                                   # decider/ ships inside the model folder
    from decider.infer import Decider
    from decider import temperature as TT
    from decider.systemone import assemble
    from decider.model import collate
    import torch

    d = Decider(path)
    name = a.name or Path(a.model.rstrip("/")).name
    bench = Path(a.bench)
    tasks = a.tasks or sorted(p.name for p in (bench / "data").iterdir() if p.is_dir())

    @torch.no_grad()
    def score(its, T):
        if d.eng is not None:
            return d.eng.score_items(its, temperature=T)
        bt = collate(its, d.m.tok.pad_token_id)
        lg = d.m.slot_logits(*[bt[k].to(d.dev) for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")])
        pr, out, c = TT.scaled_softmax(lg, TT.slot_temperatures(T, its)).float().cpu(), [], 0
        for it in its:
            out.append(pr[c:c + len(it["slots"])]); c += len(it["slots"])
        return out

    def predict(items):
        plans, flat_items, answers = [], [], {}
        for item in items:
            for qid, q in item["questions"].items():
                try:
                    rqs, index, its = d._system_one_items(item["state"], {qid: q})
                except ValueError:                             # e.g. more score levels or options than decider serves
                    answers[(item["id"], qid)] = None
                    continue
                plans.append((item["id"], qid, rqs, index, len(flat_items), len(its))); flat_items += its
        order = sorted(range(len(flat_items)), key=lambda i: len(flat_items[i]["ids"]))   # similar lengths share a pass
        T = TT.for_items(d.T, d.T_by_type, [flat_items[i] for i in order])
        flat, i = [None] * len(flat_items), 0
        while i < len(order):
            per = max(1, a.max_tokens // len(flat_items[order[min(len(order) - 1, i + 63)]]["ids"]))
            j = min(len(order), i + min(per, 64))
            for k, ps in zip(order[i:j], score([flat_items[k] for k in order[i:j]], TT.item_slice(T, i, j))):
                flat[k] = [p.tolist() for p in ps]
            i = j
        for iid, qid, rqs, index, s, n in plans:
            answers[(iid, qid)] = as_probs(assemble(rqs, index, [p for ps in flat[s:s + n] for p in ps])[qid])
        return [{"id": item["id"], "probs": {qid: answers[(item["id"], qid)] for qid in item["questions"]}} for item in items]

    print(f"{d.name} ({path}) on {len(tasks)} tasks -> results/{name}", flush=True)
    for task in tasks:
        for split in ("dev", "test"):
            src, out = bench / "data" / task / f"{split}.jsonl", bench / "results" / name / task / f"{split}.jsonl"
            if not src.exists() or out.exists():
                continue
            items, t0 = read(src), time.time()
            rows = predict(items)
            save(out, rows)
            skipped = sum(p is None for r in rows for p in r["probs"].values())
            print(f"{name} {task} {split}: {len(items)} items in {time.time() - t0:.0f}s" + (f" ({skipped} questions skipped)" if skipped else ""), flush=True)


if __name__ == "__main__":
    main()
