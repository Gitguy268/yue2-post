#!/usr/bin/env python3
"""YuE2-3B with the clear-lyrics KTO adapter: one song from a style prompt and lyrics.

The adapter is only active while YuE2 writes its semantic (music) tokens; score planning, the NAR and the VAE use
the official weights, and --adapter none gives the official model bit for bit.

SPDX-License-Identifier: Apache-2.0  (code; the adapter weights are CC BY-NC 4.0)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yue2_lora import ARLora  # noqa: E402

ADAPTER_REPO = "Lucasgitguy-co/YuE2-3B-clear-lyrics-kto-lora"


class ClearLyricsPipeline:
    """Thin wrapper around yue2.YuE2Pipeline that switches the adapter on for the semantic stage only."""

    def __init__(self, pipe):
        self.pipe, self.lora = pipe, None

    @classmethod
    def from_pretrained(cls, model="m-a-p/YuE2-3B", **kwargs):
        from yue2 import YuE2Pipeline
        return cls(YuE2Pipeline.from_pretrained(model, **kwargs))

    def use_adapter(self, path):
        if self.lora is not None:
            self.lora.detach()
            self.lora = None
        if path is not None:
            model = self.pipe._load_model()
            self.lora = ARLora.load(model, path).attach()
        return self

    def __call__(self, style, lyrics, *, cot="full", seed=831001, abc=None, cfg_scale=None):
        from contextlib import nullcontext
        from yue2.pipeline import SongResult
        import time
        p = self.pipe
        request = p._request(style, lyrics, cot=cot, seed=seed, abc=abc, cfg_scale=cfg_scale)
        config = p.effective_config(request)
        start = time.perf_counter()
        plan = p.plan(request=request)
        with self.lora.on() if self.lora is not None else nullcontext():
            semantic = p.generate_semantic(plan)
        latents = p.synthesize(semantic)
        audio = p.decode(latents)
        timing = {"e2e_seconds": time.perf_counter() - start}
        config = dict(config, adapter=ADAPTER_REPO if self.lora is not None else None)
        return SongResult(audio, 48000, semantic, latents, config, p.weights, timing, "clear-lyrics")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--style", required=True, help='e.g. "English, indie pop, acoustic guitar, warm female vocal, 100 BPM"')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lyrics"); g.add_argument("--lyrics-file", type=Path)
    ap.add_argument("--seed", type=int, default=831001)
    ap.add_argument("--cot", default="full", choices=["full", "melody", "off"])
    ap.add_argument("--adapter", default="adapter.safetensors",
                    help="path to adapter.safetensors, 'hub' to download it, or 'none' for the official model")
    ap.add_argument("--device", default="auto", help="auto picks cuda, then mps (Apple Silicon), then cpu")
    ap.add_argument("--out", type=Path, default=Path("song.flac"))
    a = ap.parse_args()
    lyrics = a.lyrics if a.lyrics is not None else a.lyrics_file.read_text(encoding="utf-8")
    lyrics = lyrics.replace("\\n", "\n")
    adapter = None
    if a.adapter == "hub":
        from huggingface_hub import hf_hub_download
        adapter = hf_hub_download(ADAPTER_REPO, "adapter.safetensors")
    elif a.adapter != "none":
        adapter = a.adapter
    import torch
    on_mac = a.device == "mps" or (a.device == "auto" and not torch.cuda.is_available()
                                   and torch.backends.mps.is_available())
    extra = dict(backend="torch-eager", memory_budget_gib=12, offload_ar=True) if on_mac else {}
    pipe = ClearLyricsPipeline.from_pretrained(device=a.device, **extra).use_adapter(adapter)
    song = pipe(a.style, lyrics, cot=a.cot, seed=a.seed)
    song.save(str(a.out))
    print(f"wrote {a.out} ({len(song.audio) / 48000:.1f} s, adapter {'on' if adapter else 'off'})")


if __name__ == "__main__":
    main()
