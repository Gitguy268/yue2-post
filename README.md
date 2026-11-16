<div align="center">

# yue2-post

Post-training for the YuE2-3B song model, aimed at clearer sung lyrics: KTO with a LoRA on the semantic stage, scored by Whisper on the vocal stem.

[Idea](#idea) · [Pipeline](#pipeline) · [Quick start](#quick-start) · [Status](#status) · [License](#license)

![Status](https://img.shields.io/badge/status-data%20generation%20in%20progress-e17055) ![Python](https://img.shields.io/badge/python-3.12-3776ab) ![Devices](https://img.shields.io/badge/devices-Apple%20MPS%20%7C%20CUDA-636e72) ![License](https://img.shields.io/badge/code-MIT%20%2B%20Apache--2.0-2d3436)

</div>

## Idea

[m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) writes full songs from a style prompt and lyrics, but the sung words are often hard to make out. This project:

1. generates songs from its own prompts and lyrics;
2. separates the vocals with Demucs and transcribes them with Whisper large-v3-turbo;
3. labels each clip by word error rate against the written lyrics (clear = desirable, garbled = undesirable);
4. trains a KTO LoRA on the AR experts, active only while YuE2 writes its semantic tokens;
5. compares adapter off and on with the same seeds on held-out prompts.

With `--adapter none` the pipeline gives the official model bit for bit, because the LoRA is applied through forward hooks.

## Pipeline

| Script | What it does |
|---|---|
| `scripts/prompts.py` | the style prompts and lyrics (162 train items, 30 held-out eval items) |
| `scripts/gen_data.py` | generates the clips (`train`, `eval-base`); resumable |
| `scripts/run_gen.sh` | runs generation with retries until all 192 clips exist |
| `scripts/score.py` | Demucs vocal stem, then Whisper WER/CER, written to `score.json` next to each clip |
| `scripts/yue2_lora.py` | the hook-based LoRA on the AR experts (q/k/v/o and MLP projections) |
| `scripts/train_kto.py` | KTO training: rank 16, alpha 32, beta 10, lr 2e-4, adapter-off model as reference |
| `scripts/yue2_clear_lyrics.py` | one song from a style and lyrics, with or without the adapter |
| `scripts/smoke_mps.py` | feasibility check of the official pipeline on Apple MPS |

## Quick start

You need Python 3.12 and either a CUDA GPU or an Apple M-series Mac (16 GB works but is slow: a 39 s clip took about 6.5 minutes on an M4).

```sh
git clone https://github.com/Gitguy268/yue2-post.git
cd yue2-post
uv venv --python 3.12 .venv
hf download m-a-p/YuE2-3B yue2_infer-0.1.5-py3-none-any.whl --local-dir .
uv pip install --python .venv ./yue2_infer-0.1.5-py3-none-any.whl
hf download m-a-p/YuE2-3B --local-dir models/YuE2-3B
hf download m-a-p/YuE2-Vae --local-dir models/YuE2-Vae
bash scripts/run_gen.sh                    # generate the clips (resumable)
```

Scoring uses a separate environment (`.venv-asr`) with `mlx-whisper`, `demucs`, `jiwer` and `soundfile`, so every clip is scored by the same ASR model:

```sh
.venv-asr/bin/python scripts/score.py
```

## Status

The official pipeline runs on MPS (peak 11.3 GiB) and scoring works. Data generation is in progress: 19 of 192 clips were done when this repository was published. The `train.jsonl` builder and the adapter-on evaluation are still missing, and the CUDA path has not been run. No adapter has been trained or published yet.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

- The repository is under the [MIT License](LICENSE).
- `scripts/yue2_lora.py` and `scripts/yue2_clear_lyrics.py` are under the [Apache License 2.0](LICENSES/Apache-2.0.txt), as their SPDX headers say. The KTO recipe follows [stevensu/YuE2-3B-instrumental-kto-lora](https://huggingface.co/stevensu/YuE2-3B-instrumental-kto-lora) (Apache-2.0 code).
- The YuE2 weights, and anything generated with them (clips, adapters), are under **CC BY-NC 4.0**: non-commercial use only. They are not in this repository.

See [NOTICE](NOTICE). This project is not affiliated with Multimodal Art Projection (m-a-p).
