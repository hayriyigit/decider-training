# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.3",
#     "transformers>=5",
#     "flash-linear-attention",
#     "peft>=0.13",
#     "huggingface_hub>=0.24",
#     "numpy>=1.24",
#     "pyarrow>=15",
#     "pyyaml>=6",
# ]
# ///
"""LoRA fine-tune of a decider model (Mapika/decider-4b) on the decision data laya_pipeline built, end to end:

  items     sample rows of laya_run/data/all/train (state / questions / gold, the LocalLLaMA/typed-decisions schema), render
            every question with decider's own prompt code, one row per question and one yes/no row per Score level (as
            decider serves them), options shuffled; hold out Turkish rows grouped by state (validation + temperatures)
  train     LoRA on attention and MLP: cross-entropy toward the gold distribution on the Turkish rows, KL toward the frozen
            base model's own answers on the English rows (replay, decider-4b v2.1's recipe against forgetting); resumable
            checkpoints, a time limit, STOP file and SIGTERM stop cleanly (exit 3: run again)
  export    merge the adapter into the bf16 weights, fit one temperature per answer type on the held-out rows
            (decider.calibrate.fit_by_type), write a loadable model folder with decider/ and decider_config.json
  evaluate  scripts/decider_eval.py on the test split: the same metrics and groups as laya_pipeline's eval_report.json
  push      export.push_to_hub: the merged model to the Hub

  uv run --script scripts/decider_lora.py --config configs/decider_lora.yaml
"""
import argparse, hashlib, json, math, os, pickle, random, shutil, signal, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

EXIT_STOPPED = 3
DEFAULTS = {
    "workdir": "./decider_run",
    "model": {"repo": "Mapika/decider-4b", "revision": "eb5fbdfc9448473ec25e399882912863afbdb70e"},
    "data": {"train": None, "test": None, "rows": {}, "holdout_rows": 1500, "max_state_tokens": 4096, "seed": 0},
    "lora": {"r": 64, "alpha": 128, "dropout": 0.0,
             "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                         "gate_proj", "up_proj", "down_proj"]},
    "train": {"epochs": 2, "lr": 1e-4, "warmup": 0.05, "tokens_per_step": 65536, "micro_tokens": 16384,
              "label_smoothing": 0.0, "seed": 0, "log_every": 20},
    "checkpoint": {"every_minutes": 30},
    "limits": {"max_hours": 0, "reserve_minutes": 20},
    "export": {"push_to_hub": None},
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def merge(base, over, where=""):
    out = dict(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise SystemExit(f"config: unknown key {where}{k}")
        out[k] = merge(base[k], v, f"{where}{k}.") if isinstance(base[k], dict) and base[k] and isinstance(v, dict) else v
    return out


def load_config(path):
    cfg = merge(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {})
    if not cfg["data"]["train"] or not cfg["data"]["rows"]:
        raise SystemExit("config: data.train (laya_pipeline's train parquet) and data.rows {source: rows} are required")
    return cfg


def key_of(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False)); tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def decider_path(cfg):
    """The base model folder; decider/ (its prompt, model and calibration code) ships inside it."""
    from huggingface_hub import snapshot_download

    m = cfg["model"]
    path = m["repo"] if Path(m["repo"]).is_dir() else snapshot_download(m["repo"], revision=m["revision"])
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


# --- items -------------------------------------------------------------------------------------------------------

def target_rows(rq, gold):
    """[(question text, options, target over the options)] of one question as decider serves it: one row, or one yes/no
    row per Score level. Laya's gold keys follow the option order: criteria keys, "0".."n-1", or false/true."""
    from decider.systemone import plan_rows

    rows, index = plan_rows({"q": rq}, True)
    p = [float(x) for x in gold["probabilities"].values()]
    if index[0][1] == "iso":
        return [(r["question"], r["options"], [1 - t, t]) for r, t in zip(rows, p)]
    return [(rows[0]["question"], rows[0]["options"], p)]


def is_english(row):
    """A replay row: trained toward the starting model's own answers. English rows, and rows marked `replay`."""
    return str(row.get("original")) == "True" or str(row.get("replay")) == "True"


def make_items(rows, tok, max_state_tokens, rng):
    from decider.infer import Q, Example
    from decider.prompt import build
    from decider.systemone import render_state, render_question

    items, skipped = [], Counter()
    for r in rows:
        ctx = render_state(json.loads(r["state"]))
        gold = json.loads(r["gold"])
        for qid, q in json.loads(r["questions"]).items():
            try:
                rq = render_question(q)
            except ValueError as e:
                skipped[str(e)[:60]] += 1
                continue
            for text, opts, target in target_rows(rq, gold[qid]):
                it = build(Example(ctx, [Q(text, list(opts), int(np.argmax(target)))], "train"), tok, rng,
                           max_options=255, max_ctx_tokens=max_state_tokens)
                perm = it["perms"][0]
                items.append({"ids": it["ids"], "slots": it["slots"], "golds": it["golds"], "nopts": it["nopts"],
                              "target": [target[i] for i in perm], "replay": is_english(r), "type": rq["type"],
                              "source": r["workflow"].split("/")[0]})
    return items, skipped


def prepare(cfg, work):
    dcfg = cfg["data"]
    key = key_of(dcfg, cfg["model"])
    meta_path = work / "items" / "meta.json"
    if meta_path.exists() and read_json(meta_path)["key"] == key:
        log(f"items: up to date ({meta_path})")
        return
    path = decider_path(cfg)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    rng = random.Random(dcfg["seed"])
    rows = pq.read_table(dcfg["train"]).to_pylist()
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["workflow"].split("/")[0]].append(r)
    unknown = set(dcfg["rows"]) - set(by_source)
    if unknown:
        raise SystemExit(f"data.rows: no such source {sorted(unknown)}; available: {sorted(by_source)}")
    picked = []
    for src, n in sorted(dcfg["rows"].items()):
        pool = by_source[src]
        picked += rng.sample(pool, min(n, len(pool)))
        log(f"  {src}: {min(n, len(pool)):,} of {len(pool):,} rows")
    # validation + temperatures: Turkish rows, whole states (a state's rows and its English twin stay on one side)
    groups = defaultdict(list)
    for r in picked:
        groups[r["group"]].append(r)
    turkish = sorted(g for g, rs in groups.items() if not any(is_english(r) for r in rs))
    rng.shuffle(turkish)
    held, n = set(), 0
    for g in turkish:
        if n >= dcfg["holdout_rows"]:
            break
        held.add(g); n += len(groups[g])
    train_rows = [r for r in picked if r["group"] not in held]
    hold_rows = [r for r in picked if r["group"] in held]
    out = {}
    for name, rs in (("train", train_rows), ("holdout", hold_rows)):
        items, skipped = make_items(rs, tok, dcfg["max_state_tokens"], rng)
        with open(work / "items" / f"{name}.pkl", "wb") as f:
            pickle.dump(items, f)
        lengths = sorted(len(it["ids"]) for it in items)
        out[name] = {"rows": len(rs), "items": len(items), "replay_items": sum(it["replay"] for it in items),
                     "tokens": sum(lengths), "tokens_p50": lengths[len(lengths) // 2], "tokens_max": lengths[-1],
                     "skipped": dict(skipped)}
        log(f"items {name}: {len(rs):,} rows -> {len(items):,} prompt rows, {sum(lengths):,} tokens"
            + (f", skipped {dict(skipped)}" if skipped else ""))
    with open(work / "items" / "holdout_rows.pkl", "wb") as f:
        pickle.dump(hold_rows, f)
    write_json(meta_path, {"key": key, **out})


# --- train -------------------------------------------------------------------------------------------------------

def micro_batches(items, max_tokens, rng):
    """Token-budgeted micro-batches of similar lengths; replay rows (KL toward the base model) batch apart, so the extra base
    forward pass runs only for them."""
    groups = defaultdict(list)
    for i, it in enumerate(items):
        n = len(it["ids"])
        groups[(it["replay"], ((n + 63) // 64) * 64)].append(i)
    out = []
    for (_, T), idx in sorted(groups.items()):
        rng.shuffle(idx)
        b = max(1, max_tokens // T)
        out += [idx[c:c + b] for c in range(0, len(idx), b)]
    rng.shuffle(out)
    return out


def steps_of(batches, items, tokens_per_step):
    """Consecutive micro-batches up to tokens_per_step padded tokens make one optimizer step."""
    steps, cur, tok = [], [], 0
    for b in batches:
        T = ((max(len(items[i]["ids"]) for i in b) + 63) // 64) * 64
        cur.append(b); tok += T * len(b)
        if tok >= tokens_per_step:
            steps.append(cur); cur, tok = [], 0
    return steps + ([cur] if cur else [])


class Stop:
    def __init__(self, work, deadline):
        self.work, self.deadline, self.signal = work, deadline, None
        for s in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(s, self._on)

    def _on(self, sig, frame):
        self.signal = signal.Signals(sig).name

    def why(self):
        if self.signal:
            return self.signal
        if (self.work / "STOP").exists():
            return "STOP file"
        if self.deadline and time.time() > self.deadline:
            return "time limit"
        return None


def build_model(cfg, path, device):
    import torch
    from decider.model import DecisionModel
    from peft import LoraConfig, get_peft_model

    model = DecisionModel(path, dtype=torch.bfloat16, grad_ckpt=False)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lc = cfg["lora"]
    # get_peft_model injects the LoRA layers into model.lm's own modules: DecisionModel.slot_logits keeps working, and
    # peft.disable_adapter() gives the base model back for the replay targets.
    peft = get_peft_model(model.lm, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                               target_modules=lc["targets"], bias="none"))
    for p in peft.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model.to(device), peft


def soft_ce(logits, target):
    import torch

    logp = torch.log_softmax(logits, -1)
    return -(target * logp.masked_fill(target == 0, 0)).sum(-1)


def batch_tensors(items, idx, pad_id, device, smoothing):
    import torch
    from decider.model import collate
    from decider.prompt import MAX_OPTIONS

    its = [items[i] for i in idx]
    b = collate(its, pad_id)
    target = torch.zeros(len(its), MAX_OPTIONS)
    for k, it in enumerate(its):
        t = torch.tensor(it["target"], dtype=torch.float32)
        if smoothing and not it["replay"]:
            t = t * (1 - smoothing) + smoothing / len(t)
        target[k, :len(t)] = t
    args = [b[k].to(device) for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")]
    return args, target.to(device), its


def save_checkpoint(work, peft, opt, progress):
    import torch
    from peft import get_peft_model_state_dict

    tmp, last = work / "checkpoints" / "tmp", work / "checkpoints" / "last"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    torch.save(get_peft_model_state_dict(peft), tmp / "adapter.pt")
    if not progress["done"]:
        torch.save(opt.state_dict(), tmp / "optimizer.pt")
    write_json(tmp / "progress.json", progress)
    shutil.rmtree(last, ignore_errors=True); tmp.rename(last)
    log(f"checkpoint: step {progress['step']:,} / {progress['steps']:,}" + (", training complete" if progress["done"] else ""))


def evaluate_holdout(model, items, pad_id, device):
    import torch

    model.eval(); correct = n = 0; loss = 0.0
    with torch.no_grad():
        for idx in micro_batches(items, 16384, random.Random(0)):
            args, target, its = batch_tensors(items, idx, pad_id, device, 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model.slot_logits(*args)
            loss += soft_ce(lg, target).sum().item()
            correct += (lg.argmax(-1) == target.argmax(-1)).sum().item(); n += len(its)
    model.train()
    return {"val_rows": n, "val_slot_accuracy": round(correct / n, 4), "val_ce": round(loss / n, 4)}


def train(cfg, work, stop):
    import torch
    from peft import set_peft_model_state_dict

    tcfg = cfg["train"]
    meta = read_json(work / "items" / "meta.json")
    tkey = key_of(meta["key"], tcfg, cfg["lora"])
    last = work / "checkpoints" / "last"
    progress = read_json(last / "progress.json") if (last / "progress.json").exists() else None
    if progress and progress["key"] != tkey:
        raise SystemExit(f"{last}: made with other data, LoRA or train settings; move it away to start over")
    if progress and progress["done"]:
        log("train: complete")
        return
    path = decider_path(cfg)
    items = pickle.loads((work / "items" / "train.pkl").read_bytes())
    holdout = pickle.loads((work / "items" / "holdout.pkl").read_bytes())
    device = "cuda"
    torch.manual_seed(tcfg["seed"])
    model, peft = build_model(cfg, path, device)
    pad_id = model.tok.pad_token_id
    params = [p for p in peft.parameters() if p.requires_grad]
    log(f"LoRA r={cfg['lora']['r']}: {sum(p.numel() for p in params) / 1e6:.1f}M trainable parameters")
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], betas=(0.9, 0.999), weight_decay=0.0)
    plans = [steps_of(micro_batches(items, tcfg["micro_tokens"], random.Random(tcfg["seed"] + e)), items,
                      tcfg["tokens_per_step"]) for e in range(tcfg["epochs"])]
    total = sum(len(p) for p in plans)
    warm = max(1, int(total * tcfg["warmup"]))
    lr_at = lambda s: tcfg["lr"] * (s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    if progress:
        set_peft_model_state_dict(peft, torch.load(last / "adapter.pt", map_location=device))
        opt.load_state_dict(torch.load(last / "optimizer.pt", map_location=device))
        log(f"resumed at step {progress['step']:,} / {total:,}")
    else:
        progress = {"key": tkey, "epoch": 0, "step_in_epoch": 0, "step": 0, "steps": total, "done": False}
    model.train()
    saved_at, t0, s0, toks = time.time(), time.time(), progress["step"], 0
    for epoch in range(progress["epoch"], tcfg["epochs"]):
        for si in range(progress["step_in_epoch"] if epoch == progress["epoch"] else 0, len(plans[epoch])):
            why = stop.why()
            if why:
                save_checkpoint(work, peft, opt, progress)
                log(f"stopped ({why}): run the same command again to resume")
                sys.exit(EXIT_STOPPED)
            step = plans[epoch][si]
            n_slots = sum(len(b) for b in step)
            for g in opt.param_groups:
                g["lr"] = lr_at(progress["step"])
            stats = defaultdict(float)
            for idx in step:
                args, target, its = batch_tensors(items, idx, pad_id, device, tcfg["label_smoothing"])
                if its[0]["replay"]:                  # the frozen base model's answer is the target (KL up to a constant)
                    with torch.no_grad(), peft.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16):
                        target = torch.softmax(model.slot_logits(*args), -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model.slot_logits(*args)
                loss = soft_ce(lg, target)
                (loss.sum() / n_slots).backward()
                kind = "kl" if its[0]["replay"] else "ce"
                stats[kind] += loss.sum().item(); stats[kind + "_n"] += len(its)
                stats["acc"] += (lg.argmax(-1) == target.argmax(-1)).sum().item()
                toks += args[0].numel()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            progress.update(step=progress["step"] + 1, epoch=epoch, step_in_epoch=si + 1)
            if progress["step"] % tcfg["log_every"] == 0:
                el = time.time() - t0; rate = (progress["step"] - s0) / el
                log(f"train epoch={epoch + 1} step={progress['step']} of={total} "
                    f"ce={stats['ce'] / max(1, stats['ce_n']):.4f} kl={stats['kl'] / max(1, stats['kl_n']):.4f} "
                    f"acc={stats['acc'] / n_slots:.3f} lr={lr_at(progress['step']):.2e} tok_per_s={toks / el:.0f} "
                    f"eta_h={(total - progress['step']) / max(rate, 1e-9) / 3600:.2f}")
            if time.time() - saved_at > cfg["checkpoint"]["every_minutes"] * 60:
                save_checkpoint(work, peft, opt, progress); saved_at = time.time()
        progress.update(epoch=epoch + 1, step_in_epoch=0)
        v = evaluate_holdout(model, holdout, pad_id, device)
        log("validation epoch_end=%d " % (epoch + 1) + " ".join(f"{k}={v[k]}" for k in v))
        with open(work / "train_log.jsonl", "a") as f:
            f.write(json.dumps({"epoch_end": epoch + 1, **v}) + "\n")
        progress["done"] = epoch + 1 == tcfg["epochs"]
        save_checkpoint(work, peft, opt, progress); saved_at = time.time()


# --- export ------------------------------------------------------------------------------------------------------

def export(cfg, work):
    import torch
    from peft import set_peft_model_state_dict

    out = work / "model"
    progress = read_json(work / "checkpoints" / "last" / "progress.json")
    if (out / "export.json").exists() and read_json(out / "export.json")["train_key"] == progress["key"]:
        log(f"export: up to date ({out})")
        return
    import gc
    gc.collect(); torch.cuda.empty_cache()                  # the training model of this process, if any
    path = decider_path(cfg)
    model, peft = build_model(cfg, path, "cpu")
    set_peft_model_state_dict(peft, torch.load(work / "checkpoints" / "last" / "adapter.pt", map_location="cpu"))
    merged = peft.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    shutil.rmtree(out, ignore_errors=True)
    merged.save_pretrained(out, safe_serialization=True)
    model.tok.save_pretrained(out)
    for name in ("decider", "chat_template.jinja", "generation_config.json"):
        src = Path(path) / name
        if src.is_dir():
            shutil.copytree(src, out / name, ignore=shutil.ignore_patterns("__pycache__"))
        elif src.exists():
            shutil.copy(src, out / name)
    del model, peft, merged
    base_cfg = read_json(Path(path) / "decider_config.json")
    write_json(out / "decider_config.json", {**base_cfg, "temperature": 1.0, "temperature_by_type": {}})

    from decider.infer import Decider
    from decider.calibrate import collect, fit_by_type

    d = Decider(str(out), use_graphs=False)
    examples = []
    for r in pickle.loads((work / "items" / "holdout_rows.pkl").read_bytes()):
        gold = json.loads(r["gold"])
        golds = {q: (g["label"] == "true" if g["type"] == "noul" else int(g["label"]) if g["type"] == "score" else g["label"])
                 for q, g in gold.items()}
        examples.append((json.loads(r["state"]), json.loads(r["questions"]), golds))
    records = []
    for ex in examples:
        try:
            records += collect(d, [ex])
        except ValueError:
            pass
    fit = fit_by_type(records)
    log(f"temperatures on {len(records):,} held-out answers: {fit['temperature_by_type']} (pooled {fit['temperature']})")
    lc, tc = cfg["lora"], cfg["train"]
    write_json(out / "decider_config.json", {
        **base_cfg, "temperature": fit["temperature"], "temperature_by_type": fit["temperature_by_type"],
        "version": f"{base_cfg.get('version', 'dev')}-tr-lora",
        "base": f"{cfg['model']['repo']}@{cfg['model']['revision']} + LoRA (merged)",
        "stage": f"LoRA rank {lc['r']} (alpha {lc['alpha']}) on {', '.join(lc['targets'])}; LR {tc['lr']}, {tc['epochs']} epochs; "
                 "cross-entropy on Turkish decision rows, KL toward the base model on English replay rows; "
                 "temperatures fitted with decider.calibrate.fit_by_type on held-out Turkish rows"})
    write_json(out / "export.json", {"train_key": progress["key"], "calibration": fit})
    log(f"exported: {out} (load with Decider({str(out)!r}))")


def evaluate(cfg, work):
    out = work / "eval_report.json"
    if out.exists() and read_json(out).get("train_key") == read_json(work / "model" / "export.json")["train_key"]:
        log(f"evaluate: up to date ({out})")
        return
    script = Path(__file__).with_name("decider_eval.py")
    tmp = work / "eval_report.tmp.json"
    subprocess.run(["uv", "run", "--script", str(script), "--data", cfg["data"]["test"], "--out", str(tmp),
                    "--model", str((work / "model").resolve())], check=True)
    rep = read_json(tmp); rep["train_key"] = read_json(work / "model" / "export.json")["train_key"]
    write_json(out, rep); tmp.unlink()
    log(f"eval: accuracy {rep['report']['all']['accuracy']}, ECE {rep['report']['all']['ece']} ({out})")


def push(cfg, work):
    from huggingface_hub import HfApi

    target = cfg["export"]["push_to_hub"]
    api = HfApi()
    api.create_repo(target["repo"], private=target.get("private", True), exist_ok=True)
    api.update_repo_settings(target["repo"], private=target.get("private", True))   # create_repo leaves an existing repo as it is
    shutil.copy(work / "eval_report.json", work / "model" / "eval_report.json")
    info = api.upload_folder(repo_id=target["repo"], folder_path=str(work / "model"), commit_message="decider LoRA fine-tune",
                             ignore_patterns=["**/__pycache__/**"])     # evaluate imported decider/ from this folder
    log(f"pushed: https://huggingface.co/{target['repo']} ({info.oid[:12]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["run", "items", "train", "export", "evaluate", "push"], default="run")
    a = ap.parse_args()
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = load_config(a.config)
    work = Path(cfg["workdir"]); (work / "items").mkdir(parents=True, exist_ok=True)
    (work / "STOP").unlink(missing_ok=True)
    lim = cfg["limits"]
    deadline = time.time() + lim["max_hours"] * 3600 - lim["reserve_minutes"] * 60 if lim["max_hours"] else None
    stop = Stop(work, deadline)
    stages = {"items": lambda: prepare(cfg, work), "train": lambda: train(cfg, work, stop), "export": lambda: export(cfg, work),
              "evaluate": lambda: evaluate(cfg, work), "push": lambda: push(cfg, work)}
    for name in (stages if a.stage == "run" else [a.stage]):
        if name == "evaluate" and not cfg["data"]["test"]:
            continue
        if name == "push" and not cfg["export"]["push_to_hub"]:
            continue
        stages[name]()
    log("done")


if __name__ == "__main__":
    main()
