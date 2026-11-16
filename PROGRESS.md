# YuE2-3B post-training — progress

Brief (user, 2026-11-07): post-train m-a-p/YuE2-3B to improve it, give run commands for MacBook Air M4 16 GB,
publish model on HF (Lucasgitguy-co) + any dataset created. Licence must stay CC-BY-NC-4.0.

## Plan
1. Feasibility gate: official wheel on MPS (py3.12 venv) + ahmadw MLX 8bit; measure speed/memory.
2. Ask user target (lyric intelligibility via ASR vs planner termination), with measured costs.
3. Self-generated data only (own prompts/lyrics), KTO LoRA on AR experts (stevensu recipe, credit Apache-2.0).
4. Held-out eval, adapter off vs on, same seed/plan, CIs.
5. Verified Mac run commands. Publish adapter (+merged?) + dataset via CLI token.

## Log
- 2026-11-08: orientation done. Not HF Pro (no Jobs) -> all compute local.
- 2026-11-08 feasibility: official wheel 0.1.5 works on MPS (backend torch-eager, offload_ar). 39 s clip = 389 s
  (abc 39 s @10 tok/s, semantic 82 s @12 tok/s, NAR 158 s, VAE 77 s), MPS peak 11.3 GiB. runs/smoke/.
- ahmadw MLX port: running its (unlicensed) code was BLOCKED by the auto-mode classifier -> not used; Mac commands use official MPS pipeline.
- Whisper turbo (mlx-whisper) on full mix hallucinates with defaults; with condition_on_previous_text=False + temp fallback
  it works. Demucs htdemucs vocals give the same transcript. Smoke clip: line 1 right, rest garbled -> real target.
- USER DECISIONS 2026-11-08: target = clearer lyrics (Whisper WER, KTO LoRA on semantic stage); run straight through
  (~160 train clips, ~30 held-out eval prompts, ~1.5 days of Mac compute).
- 2026-11-08 09:40: data generation started: scripts/gen_data.py train then eval-base, log runs/gen.log (resumable, rerun same command)
- 2026-11-08 11:01: attempt 1 died with the session (0 clips). Relaunched via scripts/launch_detached.py (own session). Disk 20 GB free.
- 2026-11-09 07:22: attempts 2-3 also died silently (0 clips; last log line "Planning score", around Claude-session restarts).
  launchd agent was refused by the auto-mode classifier (persistence). Asked the user to run scripts/run_gen.sh in their own
  Terminal window (retry loop, resumable, done when 192 item.json exist). Next: score.py (.venv-asr), build data/kto/train.jsonl,
  train_kto.py, eval adapter arm, score, publish.
- 2026-11-15: gen_data.py and train_kto.py pick cuda before mps; run_gen.sh is path-independent and uses caffeinate only on a Mac.
  Originals in runs/pre-cuda-backup/. CUDA path never run. Plan: generate on a rented GPU, score on the Mac (mlx-whisper,
  same ASR for every clip), eval-base + adapter-on arms on the same GPU. Still missing: train.jsonl builder + adapter-on eval
  generation. Rental steps: ~/RENT-GPU.md. 19/192 clips done.
