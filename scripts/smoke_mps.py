"""Feasibility gate: one short song on MPS with the official pipeline, timing each stage."""
import json, resource, sys, time
from pathlib import Path
import torch
from yue2 import YuE2Pipeline

root = Path(__file__).resolve().parents[1]
out = root / "runs" / "smoke"
out.mkdir(parents=True, exist_ok=True)

style = "English, indie pop, bright acoustic guitar, soft drums, warm female lead vocal, 100 BPM"
lyrics = """[verse]
Morning light is spilling on the kitchen floor
Coffee on the table and the radio is on
[chorus]
Sing it to me slowly, sing it to me clear
Every little word I want the world to hear
"""

t0 = time.perf_counter()
pipe = YuE2Pipeline.from_pretrained(str(root / "models/YuE2-3B"), vae=str(root / "models/YuE2-Vae"),
                                    local_files_only=True, device="mps", backend="torch-eager",
                                    memory_budget_gib=12, offload_ar=True, verify_hashes=True)
print("init", round(time.perf_counter() - t0, 1), flush=True)
mode = sys.argv[1] if len(sys.argv) > 1 else "full"
song = pipe(style=style, lyrics=lyrics, cot=mode, seed=1234)
song.save(str(out / f"smoke-{mode}.flac"))
song.save_artifacts(str(out / f"artifacts-{mode}"))
t = song.timing
report = {
    "audio_seconds": len(song.audio) / 48000 if hasattr(song, "audio") else None,
    "abc": t["abc"], "semantic": t["semantic"], "nar_seconds": t["nar_seconds"],
    "vae_seconds": t["vae_seconds"], "e2e_seconds": t["e2e_seconds"], "load": t["load"],
    "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
    "mps_driver_gib": torch.mps.driver_allocated_memory() / 2**30,
}
(out / f"timing-{mode}.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
