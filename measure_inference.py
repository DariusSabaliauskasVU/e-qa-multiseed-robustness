#!/usr/bin/env python3
"""
Inference latency and GPU memory for the ten configurations (reviewer request
R1 #14 / R2 #11). No training is needed.

Why untrained weights are sufficient: the latency and memory of a forward pass
depend on the architecture, the input length and the hardware, not on the
values of the weights. Each configuration is therefore built exactly as in
run_experiments.py (same backbone, same head, MAX_LEN = 384, DOC_STRIDE = 128,
same sliding-window features and the same constrained joint span decoding),
and RE-120 is passed through it. Prediction quality is NOT measured here.

At inference time the frozen configurations run encoder + head on every input
(the embedding cache is a training-time shortcut only), so they are measured
as encoder + head.

What is timed per question: forward pass over all windows of the question
(one batch) + transfer of logits to CPU + joint span decoding, with
torch.cuda.synchronize() before and after. Several warm-up questions are run
first and not counted.

Usage (laptop plugged in, maximum-performance power mode, other GPU programs closed):
    python measure_inference.py --config all --device cuda --repeats 5
    python measure_inference.py --config all --device cpu  --repeats 2   # optional: CPU-only deployment

Outputs:
    inference_per_run.csv   one row per configuration x repeat
    inference_summary.csv   mean and SD over repeats, one row per configuration
    inference_env.json      GPU, driver/CUDA, torch and transformers versions
"""
import argparse, csv, json, os, platform, statistics as st, time, collections

from run_experiments import (CONFIGS, prepare_eval_features, load_re120,
                             build_model, _forward_logits, decode_span, set_all_seeds)


def gpu_state():
    """SM clock, temperature, power draw and performance state, to detect throttling."""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,pstate",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return out.replace("\n", " / ")
    except Exception:
        return "nvidia-smi n/a"


def gpu_burn(max_seconds=90.0, need_p0_seconds=10.0):
    """Bring a laptop GPU out of its low-power state before any timing.
    Runs large matrix multiplications until the GPU has reported P0 for
    need_p0_seconds, or until max_seconds have passed."""
    import torch
    a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
    t0 = time.perf_counter(); p0_since = None; last = 0.0
    while time.perf_counter() - t0 < max_seconds:
        for _ in range(20):
            a = (a @ b).tanh_()
        torch.cuda.synchronize()
        now = time.perf_counter()
        if now - last >= 2.0:
            last = now; state = gpu_state()
            if "P0" in state:
                p0_since = p0_since or now
                if now - p0_since >= need_p0_seconds:
                    break
            else:
                p0_since = None
    print(f"[gpu burn-in] {time.perf_counter() - t0:.0f} s; {gpu_state()}", flush=True)
    del a, b; torch.cuda.empty_cache()


def count_params(model, cfg):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_config(name, args, device):
    import torch, numpy as np
    from transformers import AutoTokenizer
    cfg = CONFIGS[name]
    tok = AutoTokenizer.from_pretrained(cfg["backbone"], trust_remote_code=cfg.get("trust_remote_code", False))
    re120 = load_re120(args.data)
    feats = prepare_eval_features(re120, tok)
    by_q = collections.defaultdict(list)
    for f in feats:
        by_q[f["sample"]].append(f)
    qids = sorted(by_q)

    print(f"\n[{name}] building model ({cfg['backbone']}) ...", flush=True)
    set_all_seeds(args.seed)
    # determinism is needed for training, not for timing; it can select slower kernels
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    model = build_model(cfg)
    p_total, p_train = count_params(model, cfg)
    model = model.to(device).eval()

    use_cuda = device == "cuda"
    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    # pre-build tensors on the device so tokenisation and host->device copies of
    # input ids are the same for every configuration
    batches = {}
    for q in qids:
        ii = torch.tensor([f["input_ids"] for f in by_q[q]], device=device)
        am = torch.tensor([f["attention_mask"] for f in by_q[q]], device=device)
        batches[q] = (ii, am)

    def run_question(q, timing=None):
        ii, am = batches[q]
        if timing is not None:
            sync(); t0 = time.perf_counter()
        sl, el = _forward_logits(model, cfg, ii, am)
        sl = sl.float().cpu().numpy(); el = el.float().cpu().numpy()   # implies a device sync
        if timing is not None:
            timing.append((time.perf_counter() - t0) * 1000.0)
        best = ("", -1e30)
        for k, f in enumerate(by_q[q]):
            text, score = decode_span(sl[k], el[k], f["offsets"], re120["context"][q], f["seq_ids"])
            if score > best[1]:
                best = (text, score)
        return best

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        weights_mb = torch.cuda.memory_allocated() / 2**20
    else:
        weights_mb = float("nan")

    rows = []
    with torch.inference_mode():
        # warm-up (not counted): at least --warmup questions and --warmup-seconds,
        # so that GPU clocks have ramped up before timing starts
        tw = time.perf_counter(); i = 0
        while i < args.warmup or time.perf_counter() - tw < args.warmup_seconds:
            run_question(qids[i % len(qids)]); i += 1
        sync()
        print(f"[{name}] warm-up: {i} questions in {time.perf_counter() - tw:.1f} s; {gpu_state()}", flush=True)
        for r in range(args.repeats):
            times, fwd = [], []
            t_all = time.perf_counter()
            for q in qids:
                sync(); t0 = time.perf_counter()
                run_question(q, fwd)
                sync(); times.append((time.perf_counter() - t0) * 1000.0)
            total_s = time.perf_counter() - t_all
            times_sorted = sorted(times)
            p95 = times_sorted[int(round(0.95 * (len(times_sorted) - 1)))]
            rows.append(dict(
                config=name, repeat=r + 1, device=device,
                n_questions=len(qids), n_windows=len(feats),
                mean_ms_per_question=round(st.mean(times), 3),
                median_ms_per_question=round(st.median(times), 3),
                p95_ms_per_question=round(p95, 3),
                mean_forward_ms=round(st.mean(fwd), 3),
                mean_decode_ms=round(st.mean(times) - st.mean(fwd), 3),
                questions_per_second=round(len(qids) / total_s, 2),
                peak_gpu_mem_MB=round(torch.cuda.max_memory_allocated() / 2**20, 1) if use_cuda else "",
                weights_gpu_mem_MB=round(weights_mb, 1) if use_cuda else "",
                params_total_M=round(p_total / 1e6, 2),
                params_trainable_M=round(p_train / 1e6, 2),
                gpu_state_after=gpu_state() if use_cuda else "",
            ))
            print(f"[{name} r{r+1}] mean {rows[-1]['mean_ms_per_question']} ms/q "
                  f"(forward {rows[-1]['mean_forward_ms']}, decode {rows[-1]['mean_decode_ms']}), "
                  f"p95 {rows[-1]['p95_ms_per_question']} ms, peak {rows[-1]['peak_gpu_mem_MB']} MB | "
                  f"{rows[-1]['gpu_state_after']}", flush=True)
    del model, batches
    if use_cuda:
        torch.cuda.empty_cache()
    return rows


def summarise(rows):
    out = []
    keys = ["mean_forward_ms", "mean_decode_ms", "mean_ms_per_question", "median_ms_per_question", "p95_ms_per_question", "questions_per_second"]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["config"], r["device"])].append(r)
    for (c, d), rs in by.items():
        row = dict(config=c, device=d, repeats=len(rs), n_windows=rs[0]["n_windows"],
                   params_total_M=rs[0]["params_total_M"], params_trainable_M=rs[0]["params_trainable_M"])
        for k in keys:
            v = [x[k] for x in rs]
            row[k + "_mean"] = round(st.mean(v), 3)
            row[k + "_sd"] = round(st.stdev(v), 3) if len(v) > 1 else 0.0
        if rs[0]["peak_gpu_mem_MB"] != "":
            row["peak_gpu_mem_MB"] = max(x["peak_gpu_mem_MB"] for x in rs)
            row["weights_gpu_mem_MB"] = rs[0]["weights_gpu_mem_MB"]
        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)


def env_info(device):
    import torch, transformers
    info = dict(python=platform.python_version(), os=platform.platform(),
                torch=torch.__version__, transformers=transformers.__version__,
                cuda_runtime=torch.version.cuda, device=device)
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_mem_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        info["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    try:
        import subprocess
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all")
    ap.add_argument("--data", default="requirements-test-120-fixed.json")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--warmup-seconds", type=float, default=15.0)
    ap.add_argument("--burn-seconds", type=float, default=90.0,
                    help="max GPU burn-in before timing (stops earlier once P0 is stable); 0 = off")
    ap.add_argument("--seed", type=int, default=13, help="only initialises untrained heads")
    ap.add_argument("--prefix", default="inference")
    args = ap.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is not available in this Python environment")
    names = list(CONFIGS) if args.config == "all" else [c.strip() for c in args.config.split(",")]
    suffix = "" if args.device == "cuda" else "_cpu"
    json.dump(env_info(args.device), open(f"{args.prefix}_env{suffix}.json", "w"), indent=2)
    if args.device == "cuda" and args.burn_seconds > 0:
        gpu_burn(args.burn_seconds)
    rows, failed = [], {}
    for n in names:
        try:
            rows += measure_config(n, args, args.device)
        except Exception as e:                                       # keep going with the other configs
            failed[n] = f"{type(e).__name__}: {e}".splitlines()[0][:300]
            print(f"[{n}] FAILED -> {failed[n]}", flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        write_csv(f"{args.prefix}_per_run{suffix}.csv", rows)       # saved after every config
    summ = summarise(rows)
    write_csv(f"{args.prefix}_summary{suffix}.csv", summ)
    print("\nconfig        ms/question (mean±SD)   p95 ms   peak MB")
    for r in summ:
        print(f"{r['config']:<12}  {r['mean_ms_per_question_mean']:>8.2f} ± {r['mean_ms_per_question_sd']:<7.2f}"
              f"  {r['p95_ms_per_question_mean']:>8.2f}  {r.get('peak_gpu_mem_MB', '')}")
    not_p0 = sorted({r["config"] for r in rows if r.get("gpu_state_after") and "P0" not in r["gpu_state_after"]})
    if not_p0:
        print(f"\nWARNING: GPU was not in P0 after some repeats of: {', '.join(not_p0)}. "
              f"Re-run them, e.g. --config {','.join(not_p0)}")
    if failed:
        print("\nFAILED configurations (re-run with --config <name> after fixing):")
        for n, e in failed.items():
            print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
