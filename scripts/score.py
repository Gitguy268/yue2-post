"""Score clips for lyric clarity: Demucs vocal stem -> Whisper large-v3-turbo -> WER/CER vs the written lyrics.

Run with .venv-asr. usage: score.py [--watch]   (writes score.json next to each audio.flac)
"""
import json, re, sys, time, unicodedata
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import jiwer
import mlx_whisper
from demucs.apply import apply_model
from demucs.pretrained import get_model

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "clips"
ASR = "mlx-community/whisper-large-v3-turbo"
CONTRACTIONS = {"don't": "do not", "won't": "will not", "can't": "cannot", "i'm": "i am", "it's": "it is",
                "that's": "that is", "there's": "there is", "we're": "we are", "you're": "you are",
                "i'll": "i will", "we'll": "we will", "doesn't": "does not", "didn't": "did not",
                "i've": "i have", "let's": "let us", "what's": "what is", "he's": "he is", "she's": "she is"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    for k, v in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text).replace("'", "")
    return " ".join(text.split())


def main():
    watch = "--watch" in sys.argv
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sep = get_model("htdemucs").to(device).eval()
    vocals_idx = sep.sources.index("vocals")
    while True:
        todo = [d for d in sorted(CLIPS.glob("*")) if (d / "item.json").exists() and not (d / "score.json").exists()]
        for d in todo:
            item = json.loads((d / "item.json").read_text())
            audio, sr = sf.read(d / "audio.flac", dtype="float32", always_2d=True)
            assert sr == 48000
            wav = torch.from_numpy(audio.T)  # [2, T]
            import torchaudio.functional as AF
            wav = AF.resample(wav, 48000, sep.samplerate)
            ref = wav.mean(0)
            mean, std = ref.mean(), ref.std() + 1e-8
            with torch.no_grad():
                stems = apply_model(sep, ((wav - mean) / std)[None].to(device), device=device, progress=False)[0]
            vocals = (stems[vocals_idx] * std + mean).mean(0).cpu().numpy()
            vocals16 = AF.resample(torch.from_numpy(vocals), sep.samplerate, 16000).numpy().astype(np.float32)
            r = mlx_whisper.transcribe(vocals16, path_or_hf_repo=ASR, language="en",
                                       condition_on_previous_text=False, temperature=(0.0, 0.2, 0.4, 0.6),
                                       compression_ratio_threshold=2.0)
            hyp, ref_text = normalize(r["text"]), normalize(item["reference"])
            wer = jiwer.wer(ref_text, hyp) if hyp else 1.0
            cer = jiwer.cer(ref_text, hyp) if hyp else 1.0
            vocal_rms = float(np.sqrt((vocals ** 2).mean()))
            out = {"asr_model": ASR, "separator": "htdemucs", "transcript": r["text"].strip(),
                   "normalized_hypothesis": hyp, "normalized_reference": ref_text,
                   "wer": wer, "cer": cer, "vocal_rms": vocal_rms,
                   "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in r["segments"]]}
            (d / "score.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
            print(f"{d.name} wer={wer:.3f} cer={cer:.3f}", flush=True)
        if not watch:
            break
        if (ROOT / "runs" / "score.stop").exists():
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
