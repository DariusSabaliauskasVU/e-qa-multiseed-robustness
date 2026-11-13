#!/usr/bin/env python3
"""
Unified, methodology-corrected training/evaluation pipeline for
"Assessing Answer Span Detection of BERT-Based Architectures for RE QA Tasks".

One script runs all 10 configurations under matched conditions
(same tokenizer settings, MAX_LEN, sliding window, train/val split, span
decoding, and metrics), varying ONLY the model/adaptation and the random
seed. This is the controlled ablation reported in the paper.

Fixes applied vs. the original per-model scripts:
  1. RE-120 loaded from the corrected JSON (exact `answer_start` spans).
  2. Sliding-window tokenization (doc_stride + overflow) -> no answer is
     silently mislabelled as position 0.
  3. UNIFIED MAX_LEN for every config (train AND eval).
  4. ONE fixed, group-wise train/val split (grouped by SQuAD context/title,
     no leakage); only the model-init seed varies across runs.
  5. Constrained JOINT span decoding (best valid start<=end within a max
     answer length), not two independent argmaxes.
  6. Official SQuAD normalisation for EM/F1 (+ ROUGE-L as a secondary metric).

Requires a GPU for real runs. Use --smoke on CPU to validate end-to-end
plumbing on a tiny subset before spending GPU hours.
"""
import argparse, json, os, random, re, string, collections, csv, time
from typing import List, Dict, Tuple

# ----- lazy heavy imports (so --selftest works without torch) -----
def _torch():
    import torch; return torch

# =========================================================
# Config registry: the 10 configurations
# =========================================================
CONFIGS = {
    # full fine-tuning encoders
    "bert":       dict(kind="ff",   backbone="bert-base-cased",              lr=3e-5, batch=16, epochs=10),
    "distilbert": dict(kind="ff",   backbone="distilbert-base-cased",        lr=1e-5, batch=16, epochs=15),
    "spanbert":   dict(kind="ff",   backbone="SpanBERT/spanbert-base-cased", lr=3e-5, batch=16, epochs=10),
    "neobert":    dict(kind="ff_custom", backbone="chandar-lab/NeoBERT",     lr=3e-5, batch=16, epochs=10, trust_remote_code=True,
                       revision="5424c8efeea6491b151d62dee55a752165407430"),  # pinned: Hugging Face snapshot used for the paper
    "modernbert": dict(kind="ff",     backbone="answerdotai/ModernBERT-base",  lr=3e-5, batch=16, epochs=10),
    # frozen BERT + LSTM heads
    "lstm_l1":    dict(kind="frozen", backbone="bert-base-cased", lr=2e-5, batch=64, epochs=20, lstm=[768]),
    "lstm_l2":    dict(kind="frozen", backbone="bert-base-cased", lr=2e-5, batch=64, epochs=20, lstm=[768,512]),
    "lstm_l3":    dict(kind="frozen", backbone="bert-base-cased", lr=2e-5, batch=64, epochs=20, lstm=[1024,512,512]),
    "linear":     dict(kind="frozen", backbone="bert-base-cased", lr=2e-5, batch=64, epochs=20, lstm=[]),
    # partial fine-tuning: last-2 BERT layers unfrozen + LSTM-3
    "partial_ft": dict(kind="partial", backbone="bert-base-cased", lr_bert=3e-5, lr_head=1e-3,
                       batch=16, epochs=10, lstm=[1024,512,512], unfreeze_last_n=2),
}

# ----- unified constants (identical for every config) -----
MAX_LEN      = 384     # unified train + eval
DOC_STRIDE   = 128
MAX_ANS_LEN  = 40      # max answer length in tokens for joint decoding
N_BEST       = 20      # top candidates per side for joint decoding
SPLIT_SEED   = 12345   # FIXED split seed (NEVER the model seed)
VAL_FRACTION = 0.2
PATIENCE     = 2
MIN_DELTA    = 0.0

# =========================================================
# SQuAD official normalisation + metrics (pure functions)
# =========================================================
def normalize_answer(s: str) -> str:
    def rm_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def rm_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
    def white(t): return " ".join(t.split())
    return white(rm_articles(rm_punc(s.lower())))

def compute_em(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))

def compute_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split(); g = normalize_answer(gold).split()
    if len(p) == 0 or len(g) == 0:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec = same / len(p); rec = same / len(g)
    return 2 * prec * rec / (prec + rec)

def rouge_l(pred: str, gold: str) -> float:
    a = normalize_answer(pred).split(); b = normalize_answer(gold).split()
    if not a or not b:
        return float(a == b)
    # LCS length
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[len(a)][len(b)]
    prec = lcs/len(a); rec = lcs/len(b)
    return 0.0 if prec+rec == 0 else 2*prec*rec/(prec+rec)

# =========================================================
# Joint constrained span decoding (pure function)
# =========================================================
def decode_span(start_logits, end_logits, offsets, context, seq_ids,
                n_best=N_BEST, max_ans_len=MAX_ANS_LEN) -> Tuple[str, float]:
    """Return (best answer text, score) using a valid start<=end window
    restricted to context tokens and a maximum answer length."""
    import numpy as np
    s_idx = np.argsort(start_logits)[-n_best:][::-1]
    e_idx = np.argsort(end_logits)[-n_best:][::-1]
    best_text, best_score = "", -1e30
    for s in s_idx:
        if seq_ids[s] != 1 or offsets[s] is None:
            continue
        for e in e_idx:
            if seq_ids[e] != 1 or offsets[e] is None:
                continue
            if e < s or (e - s + 1) > max_ans_len:
                continue
            score = float(start_logits[s] + end_logits[e])
            if score > best_score:
                best_score = score
                best_text = context[offsets[s][0]:offsets[e][1]]
    return best_text, best_score

# =========================================================
# Grouped train/val split (pure, deterministic — independent of model seed)
# =========================================================
def grouped_split(groups: List, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """Deterministic group-wise split. Returns (train_idx, val_idx).
    All examples sharing a group id stay on the same side (no leakage).
    Falls back to a plain index split if there are fewer than 2 groups,
    and always keeps BOTH sides non-empty."""
    n = len(groups)
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        idx = list(range(n)); random.Random(seed).shuffle(idx)
        n_val = min(n-1, max(1, int(round(n*val_fraction))))
        val = set(idx[:n_val])
        return [i for i in range(n) if i not in val], [i for i in range(n) if i in val]
    rng = random.Random(seed); rng.shuffle(uniq)
    n_val = min(len(uniq)-1, max(1, int(round(len(uniq)*val_fraction))))
    val_groups = set(uniq[:n_val])
    train_idx = [i for i,g in enumerate(groups) if g not in val_groups]
    val_idx   = [i for i,g in enumerate(groups) if g in val_groups]
    return train_idx, val_idx

# The runs reported in the paper seeded random, NumPy, torch and CUDA only and used
# cuDNN with its default settings. The determinism settings below were added after
# those runs (August 2026). Use --paper-seeding to reproduce the paper's seeding exactly.
DETERMINISTIC = True

def set_all_seeds(seed: int):
    """Seed random, NumPy, torch and CUDA; unless --paper-seeding is given, also request
    deterministic cuDNN kernels. (PYTHONHASHSEED only affects child processes; set it
    in the shell before starting Python to fix str hashing.)"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = _torch()
    random.seed(seed); import numpy as np; np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not DETERMINISTIC:
        return
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def seed_worker(worker_id):
    """DataLoader worker seeding (pass as worker_init_fn to any DataLoader)."""
    import numpy as np, random as _r
    torch = _torch()
    ws = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(ws); _r.seed(ws)

# =========================================================
# Sliding-window feature preparation (train: label spans; eval: keep offsets)
# =========================================================
def prepare_train_features(examples, tokenizer):
    """examples: dict with lists 'question','context','answers'({'text','answer_start'}).
    Returns list of features with input_ids, attention_mask, start, end.
    A window not containing the answer is labelled (0,0) = CLS (no-answer)."""
    tok = tokenizer(
        examples["question"], examples["context"],
        truncation="only_second", max_length=MAX_LEN, stride=DOC_STRIDE,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tok.pop("overflow_to_sample_mapping")
    offsets = tok.pop("offset_mapping")
    feats = []
    for i in range(len(tok["input_ids"])):
        seq_ids = tok.sequence_ids(i)
        sample = sample_map[i]
        answers = examples["answers"][sample]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        off = offsets[i]
        # context token range
        cs = 0
        while seq_ids[cs] != 1: cs += 1
        ce = len(seq_ids)-1
        while seq_ids[ce] != 1: ce -= 1
        if not (off[cs][0] <= start_char and off[ce][1] >= end_char):
            s_tok = e_tok = 0            # answer not fully in this window -> CLS
        else:
            s = cs
            while s <= ce and off[s][0] <= start_char: s += 1
            e = ce
            while e >= cs and off[e][1] >= end_char: e -= 1
            s_tok, e_tok = s-1, e+1
        feats.append(dict(input_ids=tok["input_ids"][i],
                          attention_mask=tok["attention_mask"][i],
                          start=s_tok, end=e_tok))
    return feats

def prepare_eval_features(examples, tokenizer):
    tok = tokenizer(
        examples["question"], examples["context"],
        truncation="only_second", max_length=MAX_LEN, stride=DOC_STRIDE,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tok.pop("overflow_to_sample_mapping")
    feats = []
    for i in range(len(tok["input_ids"])):
        seq_ids = tok.sequence_ids(i)
        off = [o if seq_ids[k]==1 else None for k,o in enumerate(tok["offset_mapping"][i])]
        feats.append(dict(input_ids=tok["input_ids"][i],
                          attention_mask=tok["attention_mask"][i],
                          offsets=off, seq_ids=seq_ids,
                          sample=sample_map[i]))
    return feats

# =========================================================
# Data loaders
# =========================================================
def load_squad(limit: int, smoke: bool = False):
    from datasets import load_dataset
    ds = load_dataset("squad_v2")["train"]
    ex = {"question": [], "context": [], "answers": [], "title": []}
    if smoke:
        # sample a FEW answerable examples from several different titles so the
        # grouped split has >=2 groups (a real run with limit=60000 spans many titles).
        per_title = collections.defaultdict(int)
        for r in ds:
            if len(r["answers"]["answer_start"]) == 0:
                continue
            t = r["title"]
            if per_title[t] >= 5:            # cap per title to force diversity
                continue
            per_title[t] += 1
            ex["question"].append(r["question"]); ex["context"].append(r["context"])
            ex["answers"].append(r["answers"]); ex["title"].append(t)
            if len(ex["question"]) >= 60 and len(per_title) >= 6:
                break
        return ex
    n = 0
    for r in ds:
        if len(r["answers"]["answer_start"]) == 0:
            continue
        ex["question"].append(r["question"]); ex["context"].append(r["context"])
        ex["answers"].append(r["answers"]); ex["title"].append(r["title"])
        n += 1
        if limit and n >= limit: break
    return ex

def load_re120(path: str):
    data = json.load(open(path, encoding="utf-8"))
    ex = {"question": [], "context": [], "gold": []}
    for it in data:
        ex["question"].append(it["question"]); ex["context"].append(it["context"])
        ex["gold"].append(it["answers"]["text"][0])
    return ex

# =========================================================
# (Model construction, training, evaluation loops)
# Kept thin here; full bodies below in build_model / train_one / evaluate.
# =========================================================
def build_model(cfg):
    import torch, torch.nn as nn
    from transformers import AutoModel, AutoModelForQuestionAnswering
    trc = cfg.get("trust_remote_code", False)
    if cfg["kind"] == "ff":
        return AutoModelForQuestionAnswering.from_pretrained(cfg["backbone"], trust_remote_code=trc, revision=cfg.get("revision"))
    if cfg["kind"] == "ff_custom":
        # Encoders (e.g. NeoBERT) that have no AutoModelForQuestionAnswering head:
        # load the base AutoModel and add a trainable linear span head. Everything is fine-tuned.
        class QAWrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(cfg["backbone"], trust_remote_code=trc, revision=cfg.get("revision"))
                self.qa = nn.Linear(self.encoder.config.hidden_size, 2)
            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                h = getattr(out, "last_hidden_state", None)
                if h is None:
                    h = out[0] if isinstance(out, (tuple, list)) else out
                s, e = self.qa(h).split(1, dim=-1)
                return s.squeeze(-1), e.squeeze(-1)
        return QAWrap()
    # frozen / partial share an encoder + (optional) LSTM stack + span heads
    class Head(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(cfg["backbone"], trust_remote_code=trc, revision=cfg.get("revision"))
            for p in self.encoder.parameters(): p.requires_grad = False
            if cfg["kind"] == "partial" and cfg.get("unfreeze_last_n", 0) > 0:
                for layer in self.encoder.encoder.layer[-cfg["unfreeze_last_n"]:]:
                    for p in layer.parameters(): p.requires_grad = True
            hid = self.encoder.config.hidden_size
            self.lstms = nn.ModuleList(); self.drops = nn.ModuleList()
            in_sz = hid
            for h in cfg.get("lstm", []):
                self.lstms.append(nn.LSTM(in_sz, h, batch_first=True))
                self.drops.append(nn.Dropout(0.1)); in_sz = h
            self.proj = nn.Linear(in_sz, 512); self.act = nn.ReLU()
            self.start = nn.Linear(512, 1); self.end = nn.Linear(512, 1)
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            x = out.last_hidden_state
            for lstm, dp in zip(self.lstms, self.drops):
                x, _ = lstm(x); x = dp(x)
            x = self.act(self.proj(x))
            return self.start(x).squeeze(-1), self.end(x).squeeze(-1)
    return Head(cfg)

def _forward_logits(model, cfg, input_ids, attention_mask):
    if cfg["kind"] == "ff":
        o = model(input_ids=input_ids, attention_mask=attention_mask)
        return o.start_logits, o.end_logits
    return model(input_ids, attention_mask)

# ---------------------------------------------------------
# FROZEN-model fast path: precompute BERT embeddings ONCE (cached in RAM),
# then train only the LSTM/linear head per seed. Matches the original
# cached design and cuts frozen-model time from hours to minutes/seed.
# ---------------------------------------------------------
class FrozenHead:
    pass  # placeholder replaced below by _make_frozen_head (needs torch at call time)

def build_frozen_head(cfg, hidden):
    import torch.nn as nn
    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstms = nn.ModuleList(); self.drops = nn.ModuleList()
            in_sz = hidden
            for h in cfg.get("lstm", []):
                self.lstms.append(nn.LSTM(in_sz, h, batch_first=True))
                self.drops.append(nn.Dropout(0.1)); in_sz = h
            self.proj = nn.Linear(in_sz, 512); self.act = nn.ReLU()
            self.start = nn.Linear(512, 1); self.end = nn.Linear(512, 1)
        def forward(self, emb):                     # emb: [B, L, H]
            x = emb
            for lstm, dp in zip(self.lstms, self.drops):
                x, _ = lstm(x); x = dp(x)
            x = self.act(self.proj(x))
            return self.start(x).squeeze(-1), self.end(x).squeeze(-1)
    return _Head()

def encode_split(encoder, feats, device, bs=32):
    """Run the FROZEN encoder once over feats; return CPU float16 array [N, L, H]."""
    import numpy as np, torch
    N = len(feats); embs = None
    encoder.eval()
    with torch.no_grad():
        for i in range(0, N, bs):
            batch = feats[i:i+bs]
            ii = torch.tensor([f["input_ids"] for f in batch], device=device)
            am = torch.tensor([f["attention_mask"] for f in batch], device=device)
            arr = encoder(input_ids=ii, attention_mask=am).last_hidden_state.half().cpu().numpy()
            if embs is None:
                embs = np.zeros((N, arr.shape[1], arr.shape[2]), dtype=np.float16)
            embs[i:i+len(batch)] = arr
    return embs

def train_head_epoch(head, emb, feats, opt, device, bs, smoke):
    import torch, numpy as np
    idx = list(range(len(feats))); random.shuffle(idx)
    lossfn = torch.nn.CrossEntropyLoss()
    for bi, i in enumerate(range(0, len(idx), bs)):
        bidx = idx[i:i+bs]
        e = torch.from_numpy(emb[bidx].astype(np.float32)).to(device)
        s = torch.tensor([feats[j]["start"] for j in bidx], device=device)
        en = torch.tensor([feats[j]["end"] for j in bidx], device=device)
        sl, el = head(e)
        loss = (lossfn(sl, s) + lossfn(el, en)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
        if smoke and bi >= 1: break

def eval_head_loss(head, emb, feats, device, bs, smoke):
    import torch, numpy as np
    head.eval(); lossfn = torch.nn.CrossEntropyLoss(); tot, n = 0.0, 0
    with torch.no_grad():
        for bi, i in enumerate(range(0, len(feats), bs)):
            e = torch.from_numpy(emb[i:i+bs].astype(np.float32)).to(device)
            s = torch.tensor([f["start"] for f in feats[i:i+bs]], device=device)
            en = torch.tensor([f["end"] for f in feats[i:i+bs]], device=device)
            sl, el = head(e)
            tot += ((lossfn(sl, s) + lossfn(el, en)) / 2).item(); n += 1
            if smoke and bi >= 1: break
    return tot / max(1, n)

def evaluate_re120_frozen(head, eval_emb, eval_feats, re120, device, bs):
    import torch, numpy as np
    head.eval()
    per_q = collections.defaultdict(lambda: ("", -1e30))
    with torch.no_grad():
        for i in range(0, len(eval_feats), bs):
            batch = eval_feats[i:i+bs]
            e = torch.from_numpy(eval_emb[i:i+len(batch)].astype(np.float32)).to(device)
            sl, el = head(e); sl = sl.cpu().numpy(); el = el.cpu().numpy()
            for k, f in enumerate(batch):
                q = f["sample"]
                text, score = decode_span(sl[k], el[k], f["offsets"], re120["context"][q], f["seq_ids"])
                if score > per_q[q][1]:
                    per_q[q] = (text, score)
    em = f1 = rg = 0.0; Nq = len(re120["gold"])
    LAST_PREDS[:] = [per_q[q][0] for q in range(Nq)]
    for q in range(Nq):
        pred = per_q[q][0]; gold = re120["gold"][q]
        em += compute_em(pred, gold); f1 += compute_f1(pred, gold); rg += rouge_l(pred, gold)
    return em/Nq, f1/Nq, rg/Nq

# Optional per-question prediction dump (--save-preds DIR). It does not change training
# or scoring; it only stores the predictions that the reported metrics are computed from.
LAST_PREDS = []
PRED_DIR = None

def save_predictions(config_name, seed, re120, row):
    if not PRED_DIR:
        return
    os.makedirs(PRED_DIR, exist_ok=True)
    items = [dict(index=q, prediction=LAST_PREDS[q], gold=re120["gold"][q]) for q in range(len(re120["gold"]))]
    with open(os.path.join(PRED_DIR, f"{config_name}_seed{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(config=config_name, seed=seed, metrics=row, items=items), fh, ensure_ascii=False, indent=1)

def append_row(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config","seed","epochs_at_convergence","train_minutes","EM","F1","ROUGE_L"])
        if not exists: w.writeheader()
        w.writerow(row)

# training / evaluation bodies live in run_config(); omitted from --selftest path.
def run_config(config_name, seeds, squad_limit, re120_path, out_csv, smoke=False):
    import torch, numpy as np
    from transformers import AutoTokenizer, AutoModel
    cfg = CONFIGS[config_name]
    tok = AutoTokenizer.from_pretrained(cfg["backbone"], trust_remote_code=cfg.get("trust_remote_code", False),
                                        revision=cfg.get("revision"))
    squad = load_squad(squad_limit, smoke=smoke)
    re120 = load_re120(re120_path)
    tr_idx, va_idx = grouped_split(squad["title"])
    def subset(ex, idx):
        return {k: [ex[k][i] for i in idx] for k in ex}
    train_feats = prepare_train_features(subset(squad, tr_idx), tok)
    val_feats   = prepare_train_features(subset(squad, va_idx), tok)
    eval_feats  = prepare_eval_features(re120, tok)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_seeds = [seeds[0]] if smoke else seeds
    rows = []

    # ---------- FROZEN fast path: encode ONCE, train head per seed ----------
    if cfg["kind"] == "frozen":
        encoder = AutoModel.from_pretrained(cfg["backbone"]).to(device).eval()
        for p in encoder.parameters(): p.requires_grad = False
        print(f"[{config_name}] caching frozen embeddings ({len(train_feats)} train / {len(val_feats)} val features)...")
        te0 = time.time()
        train_emb = encode_split(encoder, train_feats, device)
        val_emb   = encode_split(encoder, val_feats, device)
        eval_emb  = encode_split(encoder, eval_feats, device)
        hidden = train_emb.shape[2]
        del encoder
        if device == "cuda": torch.cuda.empty_cache()
        print(f"[{config_name}] embeddings cached in {(time.time()-te0)/60:.1f} min; hidden={hidden}")
        for seed in use_seeds:
            set_all_seeds(seed)
            head = build_frozen_head(cfg, hidden).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"])
            t0 = time.time(); best_val, bad, best_state, epoch_done = 1e30, 0, None, 0
            for epoch in range(1 if smoke else cfg["epochs"]):
                head.train(); train_head_epoch(head, train_emb, train_feats, opt, device, cfg["batch"], smoke)
                vloss = eval_head_loss(head, val_emb, val_feats, device, cfg["batch"], smoke)
                epoch_done = epoch + 1
                if vloss < best_val - MIN_DELTA:
                    best_val, bad = vloss, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE: break
            if best_state: head.load_state_dict(best_state)
            minutes = (time.time() - t0) / 60.0
            em, f1, rg = evaluate_re120_frozen(head, eval_emb, eval_feats, re120, device, cfg["batch"])
            row = dict(config=config_name, seed=seed, epochs_at_convergence=epoch_done,
                       train_minutes=round(minutes, 2), EM=round(em, 4), F1=round(f1, 4), ROUGE_L=round(rg, 4))
            rows.append(row); append_row(out_csv, row)          # <-- per-seed write
            save_predictions(config_name, seed, re120, row)
            print(f"[{config_name} seed={seed}] EM={em:.4f} F1={f1:.4f} ROUGE-L={rg:.4f} ({minutes:.1f} min)")
        return rows

    # ---------- FULL-FT / PARTIAL path: train the encoder end-to-end ----------
    for seed in use_seeds:
        set_all_seeds(seed)
        model = build_model(cfg).to(device)
        if cfg["kind"] == "partial":
            opt = torch.optim.AdamW([
                {"params": [p for n,p in model.named_parameters() if p.requires_grad and n.startswith("encoder")], "lr": cfg["lr_bert"]},
                {"params": [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("encoder")], "lr": cfg["lr_head"]},
            ])
        else:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
        t0 = time.time(); best_val, bad, best_state, epoch_done = 1e30, 0, None, 0
        for epoch in range(1 if smoke else cfg["epochs"]):
            model.train(); train_epoch(model, cfg, train_feats, opt, device, cfg["batch"], smoke)
            vloss = eval_loss(model, cfg, val_feats, device, cfg["batch"], smoke)
            epoch_done = epoch + 1
            if vloss < best_val - MIN_DELTA:
                best_val, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE: break
        if best_state: model.load_state_dict(best_state)
        minutes = (time.time() - t0) / 60.0
        em, f1, rg = evaluate_re120(model, cfg, eval_feats, re120, device, cfg["batch"])
        row = dict(config=config_name, seed=seed, epochs_at_convergence=epoch_done,
                   train_minutes=round(minutes, 2), EM=round(em, 4), F1=round(f1, 4), ROUGE_L=round(rg, 4))
        rows.append(row); append_row(out_csv, row)              # <-- per-seed write
        save_predictions(config_name, seed, re120, row)
        print(f"[{config_name} seed={seed}] EM={em:.4f} F1={f1:.4f} ROUGE-L={rg:.4f} ({minutes:.1f} min)")
    return rows

def _batches(feats, bs):
    for i in range(0, len(feats), bs):
        yield feats[i:i+bs]

def train_epoch(model, cfg, feats, opt, device, bs, smoke):
    import torch
    random.shuffle(feats)
    lossfn = torch.nn.CrossEntropyLoss()
    for bi, batch in enumerate(_batches(feats, bs)):
        ii = torch.tensor([f["input_ids"] for f in batch], device=device)
        am = torch.tensor([f["attention_mask"] for f in batch], device=device)
        s  = torch.tensor([f["start"] for f in batch], device=device)
        e  = torch.tensor([f["end"] for f in batch], device=device)
        sl, el = _forward_logits(model, cfg, ii, am)
        loss = (lossfn(sl, s) + lossfn(el, e)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
        if smoke and bi >= 1: break

def eval_loss(model, cfg, feats, device, bs, smoke):
    import torch
    model.eval(); lossfn = torch.nn.CrossEntropyLoss(); tot, n = 0.0, 0
    with torch.no_grad():
        for bi, batch in enumerate(_batches(feats, bs)):
            ii = torch.tensor([f["input_ids"] for f in batch], device=device)
            am = torch.tensor([f["attention_mask"] for f in batch], device=device)
            s  = torch.tensor([f["start"] for f in batch], device=device)
            e  = torch.tensor([f["end"] for f in batch], device=device)
            sl, el = _forward_logits(model, cfg, ii, am)
            tot += ((lossfn(sl, s) + lossfn(el, e)) / 2).item(); n += 1
            if smoke and bi >= 1: break
    return tot / max(1, n)

def evaluate_re120(model, cfg, eval_feats, re120, device, bs):
    """Aggregate windows per question, pick global best span, score EM/F1/ROUGE-L."""
    import torch, numpy as np
    model.eval()
    per_q = collections.defaultdict(lambda: ("", -1e30))
    with torch.no_grad():
        for batch in _batches(eval_feats, bs):
            ii = torch.tensor([f["input_ids"] for f in batch], device=device)
            am = torch.tensor([f["attention_mask"] for f in batch], device=device)
            sl, el = _forward_logits(model, cfg, ii, am)
            sl = sl.cpu().numpy(); el = el.cpu().numpy()
            for k, f in enumerate(batch):
                q = f["sample"]
                text, score = decode_span(sl[k], el[k], f["offsets"], re120["context"][q], f["seq_ids"])
                if score > per_q[q][1]:
                    per_q[q] = (text, score)
    em = f1 = rg = 0.0; N = len(re120["gold"])
    LAST_PREDS[:] = [per_q[q][0] for q in range(N)]
    for q in range(N):
        pred = per_q[q][0]; gold = re120["gold"][q]
        em += compute_em(pred, gold); f1 += compute_f1(pred, gold); rg += rouge_l(pred, gold)
    return em/N, f1/N, rg/N

def write_rows(path, rows):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config","seed","epochs_at_convergence","train_minutes","EM","F1","ROUGE_L"])
        if not exists: w.writeheader()
        w.writerows(rows)

# =========================================================
# Self-test of the correctness-critical PURE functions (no torch/models)
# =========================================================
def selftest():
    # metrics
    assert compute_em("the cat", "The Cat!") == 1
    assert compute_em("cat", "dog") == 0
    # pred normalises to "quick brown fox" (3 words), gold "quick brown" (2): prec 2/3, rec 1 -> F1 0.8
    assert abs(compute_f1("the quick brown fox", "quick brown") - 0.8) < 1e-9
    assert abs(rouge_l("a b c d", "a b c d") - 1.0) < 1e-9
    # grouped split determinism + no leakage
    groups = [g for g in range(20) for _ in range(3)]
    a1 = grouped_split(groups); a2 = grouped_split(groups)
    assert a1 == a2, "split not deterministic"
    tr, va = a1
    trg = {groups[i] for i in tr}; vg = {groups[i] for i in va}
    assert trg.isdisjoint(vg), "group leakage between train and val"
    assert set(tr).isdisjoint(set(va))
    # both sides non-empty even in degenerate cases (single group / two groups)
    for g in (["A"]*10, ["A"]*5+["B"]*5):
        t2, v2 = grouped_split(g)
        assert len(t2) > 0 and len(v2) > 0, "empty split side"
    # joint decode: start>end must never be returned; picks best valid window
    ctx = "The system shall compress large image files automatically."
    # fake offsets: one token per word (context side)
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", ctx)]
    offs = [None] + [(w[1], w[2]) for w in words] + [None]
    seq  = [0] + [1]*len(words) + [0]
    # offs index: 1=The 2=system 3=shall 4=compress 5=large 6=image 7=files 8=automatically.
    sl = [ -9.0 ]*len(seq); el = [ -9.0 ]*len(seq)
    sl[4] = 5.0   # start at "compress"
    el[6] = 5.0   # valid end at "image"
    el[2] = 9.0   # higher end BEFORE the start -> must be rejected by the constraint
    text, _ = decode_span(sl, el, offs, ctx, seq)
    assert text == "compress large image", f"joint decode wrong: {text!r}"
    # answer must be a contiguous context substring
    assert text in ctx
    print("SELF-TEST PASSED: metrics, grouped split (no leakage), joint span decoding.")

# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all", help="'all' or comma-separated config names (e.g. neobert,lstm_l1)")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[13,42,73,101,123,202,333,777,1337,2024])
    ap.add_argument("--squad-limit", type=int, default=60000)
    ap.add_argument("--data", default="requirements-test-120-fixed.json")
    ap.add_argument("--out", default="results_per_seed.csv")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU end-to-end check")
    ap.add_argument("--selftest", action="store_true", help="test pure functions only (no torch)")
    ap.add_argument("--paper-seeding", action="store_true",
                    help="seed exactly as in the runs reported in the paper (no cuDNN determinism flags)")
    ap.add_argument("--save-preds", default=None, metavar="DIR",
                    help="also write per-question predictions to DIR/<config>_seed<seed>.json")
    args = ap.parse_args()
    global DETERMINISTIC, PRED_DIR
    PRED_DIR = args.save_preds
    if args.paper_seeding:
        DETERMINISTIC = False
    if args.selftest:
        selftest(); return
    if args.config == "all":
        names = list(CONFIGS)
    else:
        names = [c.strip() for c in args.config.split(",") if c.strip()]
        bad = [n for n in names if n not in CONFIGS]
        if bad:
            ap.error(f"unknown config(s): {bad}. valid: {list(CONFIGS)}")
    for n in names:
        run_config(n, args.seeds, args.squad_limit, args.data, args.out, smoke=args.smoke)

if __name__ == "__main__":
    main()
