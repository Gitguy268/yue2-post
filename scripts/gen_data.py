"""Generate clips with the base YuE2-3B on MPS. Resumable: finished items are skipped.

usage: gen_data.py train|eval-base
Writes data/clips/<id>/{audio.flac, item.json, abc_ids.npy, semantic.npy, prefix.npy, timing.json}
"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from yue2 import YuE2Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from prompts import train_items, eval_items

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "clips"
# NVIDIA (rented box) first, then Apple Metal
DEVICE = "cuda" if torch.cuda.is_available() else "mps"


def load_pipe():
    return YuE2Pipeline.from_pretrained(str(ROOT / "models/YuE2-3B"), vae=str(ROOT / "models/YuE2-Vae"),
                                        local_files_only=True, device=DEVICE, backend="torch-eager",
                                        memory_budget_gib=12, offload_ar=True, verify_hashes=False,
                                        progress=True)


def render(pipe, item, out, semantic_hook=None):
    """plan -> semantic -> NAR -> VAE; semantic_hook is a context manager factory (adapter on)."""
    from contextlib import nullcontext
    t0 = time.perf_counter()
    plan = pipe.plan(style=item["style"], lyrics=item["lyrics"], cot="full", seed=item["seed"])
    with (semantic_hook() if semantic_hook else nullcontext()):
        sem = pipe.generate_semantic(plan)
    t1 = time.perf_counter()
    latents = pipe.synthesize(sem)
    t2 = time.perf_counter()
    audio = pipe.decode(latents)
    t3 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "audio.flac", audio, 48000)
    np.save(out / "abc_ids.npy", np.array(plan.abc_ids, dtype=np.int32))
    np.save(out / "prefix.npy", np.array(plan.prefix, dtype=np.int32))
    np.save(out / "semantic.npy", np.array(sem.tokens, dtype=np.int32))
    (out / "score.abc").write_text(plan.abc or "")
    timing = {"plan_and_semantic": t1 - t0, "nar": t2 - t1, "vae": t3 - t2, "total": t3 - t0,
              "audio_seconds": len(audio) / 48000, "abc_tokens": len(plan.abc_ids),
              "semantic_tokens": len(sem.tokens), "plan_truncated": plan.truncated,
              "semantic_truncated": sem.truncated}
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    (out / "item.json").write_text(json.dumps(item, indent=2, ensure_ascii=False))  # written last = done marker
    return timing


def main():
    which = sys.argv[1]
    items = train_items() if which == "train" else eval_items()
    suffix = "" if which == "train" else "-base"
    torch.manual_seed(0)
    pipe = load_pipe()
    for n, item in enumerate(items):
        out = CLIPS / (item["id"] + suffix)
        if (out / "item.json").exists():
            continue
        try:
            t = render(pipe, item, out)
            print(f"[{n + 1}/{len(items)}] {out.name} audio={t['audio_seconds']:.1f}s total={t['total']:.0f}s "
                  f"trunc={t['plan_truncated'] or t['semantic_truncated']}", flush=True)
        except Exception:
            print(f"[{n + 1}/{len(items)}] {out.name} FAILED", flush=True)
            traceback.print_exc()
    print("DONE", which, flush=True)


if __name__ == "__main__":
    main()
