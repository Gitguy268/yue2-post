"""KTO on YuE2-3B's semantic stage with a LoRA on the AR experts, on MPS.

Recipe follows stevensu/YuE2-3B-instrumental-kto-lora (Apache-2.0 code, CC-BY-NC weights): rank 16 / alpha 32 on the AR
experts, mean log-probability over the semantic tokens + end token, beta 10, lr 2e-4, adapter-off model as reference.
Our labels: Whisper WER on the Demucs vocal stem (desirable = clearly sung, undesirable = garbled).

usage: train_kto.py data/kto/train.jsonl runs/kto [--epochs 4]
"""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import sys
sys.path.insert(0, str(Path(__file__).parent))
from yue2_lora import ARLora
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END

ROOT = Path(__file__).resolve().parents[1]
# NVIDIA (rented box) first, then Apple Metal
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps")


def load_ar_model():
    model = YuE2ForCausalLM.from_pretrained(ROOT / "models/YuE2-3B", local_files_only=True,
                                            dtype=torch.bfloat16, low_cpu_mem_usage=True)
    for layer in model.model.layers:  # AR-only training: drop the NAR experts (~2.9 GB)
        for name in ("nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp"):
            if hasattr(layer, name):
                delattr(layer, name)
    for name in ("llm2vae", "vae2llm", "time_embedder", "latent_pos_embed"):
        delattr(model, name)
    model.eval().requires_grad_(False)
    return model.to(DEV)


def sequence(rec):
    d = ROOT / rec["clip_dir"]
    prefix = np.load(d / "prefix.npy").astype(np.int64).tolist()
    sem = np.load(d / "semantic.npy").astype(np.int64)
    targets = (sem + CODEC_OFFSET).tolist() + [MUSIC_END]  # generated tokens incl. end
    return prefix, targets


def mean_logprob(model, prefix, targets, grad):
    ids = torch.tensor([prefix + targets[:-1]], device=DEV)
    pos = torch.arange(ids.shape[1], device=DEV)[None]
    bb = model.model
    x = bb.embed_tokens(ids)
    cos, sin = bb.rotary_emb(pos)
    for layer in bb.layers:
        if grad:
            x = checkpoint(lambda h, l=layer: l(h, cos, sin, None, layer_idx=0), x, use_reentrant=False)
        else:
            x = layer(x, cos, sin, None, layer_idx=0)
    h = bb.norm(x)[0, len(prefix) - 1:]  # positions predicting each target
    # Sampling only ever allows codec tokens + MUSIC_END, so normalise over that set.
    w = model.lm_head.weight
    rows = torch.cat([w[CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE], w[MUSIC_END:MUSIC_END + 1]])
    logits = (h @ rows.t()).float()
    tgt = torch.tensor(targets, device=DEV)
    local = torch.where(tgt == MUSIC_END, torch.full_like(tgt, CODEC_SIZE), tgt - CODEC_OFFSET)
    lp = torch.log_softmax(logits, -1).gather(-1, local[:, None])[:, 0]
    return lp.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta", type=float, default=10.0); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    recs = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    n_d = sum(r["label"] for r in recs); n_u = len(recs) - n_d
    # KTO class weights: lambda_D * n_D ~= lambda_U * n_U
    lam_d, lam_u = (1.0, n_d / max(n_u, 1)) if n_d <= n_u else (n_u / max(n_d, 1), 1.0)
    print(f"{len(recs)} examples: {n_d} desirable, {n_u} undesirable; lambda_D={lam_d:.3f} lambda_U={lam_u:.3f}", flush=True)

    model = load_ar_model()
    lora = ARLora(model, rank=args.rank, alpha=args.alpha).attach()

    ref_path = out / "ref_logps.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    with torch.no_grad():
        for r in recs:
            if r["id"] not in ref:
                ref[r["id"]] = mean_logprob(model, *sequence(r), grad=False).item()
                ref_path.write_text(json.dumps(ref))
    print("reference log-probs ready", flush=True)

    opt = torch.optim.AdamW(lora.parameters(), lr=args.lr, weight_decay=0.0)
    total_updates = math.ceil(len(recs) / args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / total_updates))
    z_ema, update, log = 0.0, 0, (out / "train_log.jsonl").open("a")
    t0 = time.time()
    for epoch in range(args.epochs):
        order = recs[:]; random.shuffle(order)
        for i, r in enumerate(order):
            with lora.on():
                lp = mean_logprob(model, *sequence(r), grad=True)
            reward = lp - ref[r["id"]]
            z0 = max(z_ema, 0.0)  # KL reference point (detached running estimate)
            if r["label"]:
                loss = lam_d * (1 - torch.sigmoid(args.beta * (reward - z0)))
            else:
                loss = lam_u * (1 - torch.sigmoid(args.beta * (z0 - reward)))
            (loss / args.accum).backward()
            z_ema = 0.9 * z_ema + 0.1 * reward.item()
            log.write(json.dumps({"epoch": epoch, "step": i, "id": r["id"], "label": r["label"],
                                  "reward": reward.item(), "z0": z0, "loss": loss.item()}) + "\n"); log.flush()
            if (i + 1) % args.accum == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); update += 1
                print(f"epoch {epoch} update {update}/{total_updates} reward={reward.item():+.4f} z0={z0:.4f} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
        lora.save(out / f"adapter-epoch{epoch + 1}.safetensors", {"epoch": epoch + 1, "beta": args.beta, "lr": args.lr})
    lora.save(out / "adapter.safetensors", {"epochs": args.epochs, "beta": args.beta, "lr": args.lr,
                                            "examples": len(recs), "desirable": n_d, "undesirable": n_u})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
