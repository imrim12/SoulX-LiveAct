# SoulX-LiveAct — Run Guide (hard-won, from a full end-to-end smoke test)

This is a **fork with fixes** of [Soul-AILab/SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct):
diffusion-based **whole-body / whole-head real-time human animation** — generates avatar video that
lip-syncs + emotes from a **reference image + driving audio + text prompt**. Paper claims **20 FPS
(real-time) on 2× H100** at 416×720 / 512×512.

> **⚠️ The upstream repo does NOT work out of the box** — a fresh install per the README produces
> **noise or solid-black (NaN) video** on H100/A100-class GPUs. This is caused by **SageAttention**
> (see fixes below). Every fix needed to get correct output is already applied in this fork. Read
> the "Fixes applied" and "Gotchas" sections before running.

---

## ✅ TL;DR — the command that actually works

Once the environment is built and weights are downloaded (see below), on **2× H100 (or 2× GPU with
≥80 GB each)**:

```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29550 \
  generate.py \
  --size 416*720 \
  --ckpt_dir models/LiveAct \
  --wav2vec_dir models/chinese-wav2vec2-base \
  --fps 20 \
  --input_json examples/smoke_all.json \
  --dura_print
```

Output: one `<image>_<audio>.mp4` per entry in the input JSON, written to the repo root.
**Always extract and *look at* a frame before trusting a run** (`rc=0` is NOT proof of correct output
— see gotchas): `ffmpeg -ss 1 -i out.mp4 -vframes 1 f.png` — a ~500 KB PNG is real; a ~1 KB PNG is a
black/NaN frame.

Input JSON format (`examples/smoke_all.json`):
```json
[{ "prompt": "A cheerful woman talking to camera.",
   "cond_image": "examples/image/5.png",
   "cond_audio": "examples/audio/5.wav" }]
```

---

## 🩹 Fixes applied in this fork (vs upstream) — and why

| Commit | Fix | Why |
|---|---|---|
| `3aebd53` | **Disable SageAttention** — `USE_SAGEATTN = False` in `model_liveact/model_memory.py` + `model_memory_sp.py`; and `xFuserLongContextAttention` attn_type `SAGE_FP8_SM90` → **`AttnType.FA`** | SageAttention **v2.2.0 produces NaN/noise** with this model on H100/A100. The model auto-enables it whenever the package is importable. Falling back to PyTorch **SDPA** fixes output. (Upstream issues #4, #12.) |
| `76e63c7` | `generate.py`: `low_cpu_mem_usage=True` in `WanModel.from_pretrained` | With `False`, the 18B model ~doubles memory during load and **OOMs a single 80 GB GPU** just loading. |
| `f15c7c7` | Dockerfile: **install flash-attn 2.7.3** | `wan/modules/attention.py` hard-asserts `FLASH_ATTN_2_AVAILABLE` but flash-attn is in *neither* `requirements.txt` nor the Dockerfile → crash. Must be `>=2.7.1,<=2.8.2` (xformers rejects 2.8.3+). Prebuilt wheel `2.7.3+cu12torch2.8cxx11abiTRUE-cp311` works on Hopper sm_90 **and** Ada sm_89 **and** Blackwell sm_120. |

**Keep `sageattention` installed** even though we don't use it — `xfuser` (2-GPU sequence-parallel
path) hard-imports it at load time; uninstalling it crashes the multi-GPU path.

---

## 🏗️ Environment setup

Base: **torch 2.8 / CUDA 12.8 / Python 3.11** (e.g. `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`).
Known-good dependency set (from upstream #12's working `pip freeze`):

```
torch==2.8.0  torchaudio==2.8.0  torchvision==0.23.0
diffusers==0.38.0  transformers==4.57.3  tokenizers==0.22.2  accelerate==1.13.0
flash-attn==2.7.3  vllm==0.11.0  xfuser==0.4.5  numpy==2.2.6
sageattention==2.2.0  (installed, but disabled — see fixes)
lightx2v (--no-deps) + av
```

Build steps (what the `Dockerfile` encodes — run these directly if you can't use Docker):
1. `pip install -r requirements.txt`
2. Build **SageAttention** v2.2.0 from source (`git clone --branch v2.2.0 … && python setup.py install`) — GPU present at runtime auto-detects arch; no `TORCH_CUDA_ARCH_LIST` needed unless building without a GPU (`docker build`).
3. Install **flash-attn 2.7.3** prebuilt wheel (see fixes table).
4. `pip install vllm==0.11.0`
5. Install **LightX2V** with `pip install --no-deps <cloned dir>` + `pip install av` (its `pyproject.toml` pulls a huge serving stack otherwise; also patch `license = "Apache-2.0"` → `license = {text = "Apache-2.0"}` for strict setuptools).
6. `pip install "diffusers==0.38.0" "transformers==4.57.3"` (pin these — see gotchas).

### 🤗 Weights (~52.5 GB)
- `Soul-AILab/LiveAct` → `models/LiveAct/` (~51 GB: 4 DiT shards + 11 GB T5 umt5-xxl + 4.8 GB CLIP + 0.5 GB Wan2.1 VAE + tokenizers)
- `TencentGameMate/chinese-wav2vec2-base` → `models/chinese-wav2vec2-base/` (~1.5 GB)

**Use `aria2c`, not `hf download`**, on flaky/throttled hosts — `hf download` *discards* nearly-complete
multi-GB shards on a dropped CDN connection and can loop forever. aria2 range-resumes:
```bash
aria2c -c -x16 -s16 -j3 --max-tries=0 --retry-wait=5 --timeout=120 \
  --max-connection-per-server=16 --min-split-size=10M --file-allocation=none -i urls.txt
```

---

## 🐛 Gotchas & issues you WILL hit (each cost hours to find)

1. **Noise / black output = SageAttention.** The #1 trap. If frames are noise or solid black, SageAttention
   is on. Verify `USE_SAGEATTN = False` in *both* `model_memory.py` and `model_memory_sp.py`, and
   `AttnType.FA` (not `SAGE_FP8_SM90`) in `model_memory_sp.py`. (Also: `attention.py` never imports
   `logging`, so its own `USE_SAGEATTN` try/except silently forces False — a separate upstream bug.)
2. **`rc=0` / "generated successfully" does NOT mean correct output.** The pipeline happily writes
   NaN→black video. **Always visually inspect a frame.** A `RuntimeWarning: invalid value encountered
   in cast` in the log = NaN in the frames.
3. **flash-attn version window.** Need `>=2.7.1,<=2.8.2`; `2.8.3` is rejected by xformers at import.
   Use `2.7.3`. (Removing xformers also works — upstream #12 — but 2.7.3 avoids that.)
4. **transformers too old breaks lightx2v.** `generate.py`→lightx2v imports
   `Gemma3ForConditionalGeneration` (transformers ≥4.50). The repo's `requirements.txt` says
   `transformers>=4.49.0` — a *fresh* install pulls latest (fine), but **do not pin an old version**.
5. **Single-GPU bf16 OOMs on load** even on 80 GB (fp32-ish materialization). Multi-GPU (seq-parallel)
   loads fine (~72 GB/GPU). For 1-GPU you need fp8 (`--fp8_gemm`) or ≥1 more GPU.
6. **Leftover `torchrun` workers pin GPU memory** and cause phantom OOMs on the next run. Between runs:
   `for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done`.
7. **MP3 header lies.** Some `.mp3` files report a long duration in their container header but contain
   far less actual audio; the model correctly matches the *real* audio length. Verify with
   `torchaudio.load`, not `ffprobe format=duration`.
8. **First run is slow (`torch.compile` warm-up)** — blocks 0–2 take ~25–45 s each (~77 s total), then
   steady state. Amortizes to zero for a long-running 24/7 service.
9. **`--fp8_gemm` (fp8 *weights*) produces garbage** on this stack (separate from SageAttention). Use
   bf16 + SDPA for correct output.

---

## 📊 Measured performance & cost (2× H100 SXM 80 GB, bf16 + SDPA, fps 20, 416×720)

| Metric | Value |
|---|---|
| Model load (2-GPU) | ~15 s |
| `torch.compile` warm-up | ~77 s (one-time; blocks 0–2) |
| **Steady-state** | ~2.2 s / 1.6 s-video block ⇒ **~0.73× real-time (~14.5 FPS)** |
| GPU telemetry | both GPUs 100% util, ~72 GB/80 GB, ~650–690 W each |
| Instance cost (Vast) | ~$4.84/hr for 2× H100 |
| **$ per hour of video** | **~$6.7** (GPU only, at 0.73× RT) |

**Reality check vs the paper's "20 FPS on 2×H100":** that figure needs SageAttention's fp8 attention +
fp8 GEMM + fusion — the exact stack that's broken here. Correct output currently forces the slower SDPA
path (**0.73× real-time, i.e. sub-real-time**). **Open path to real-time + lower cost:** upstream #4
suggests **updating SageAttention to the latest version** (not the pinned v2.2.0) may restore *both*
correctness and speed — **untested, highest-value next experiment.** Also untested: `--steam_audio`
(streaming-audio mode) and fp8 combined with SDPA.

---

## 🖥️ Running on Vast.ai (no Docker-in-Docker)

Vast instances are themselves containers → **can't run `docker compose`**. Run the Dockerfile's steps
directly over SSH (that's exactly what the smoke-test `setup.sh` does). On a Docker-capable host,
`docker compose build && docker compose run --rm generate` works — but you must download the 52 GB
weights into `models/` first (the compose mounts them read-only; it does not download them).

Config recipes (from upstream README, all need the SageAttention fix):
- **Real-time, 2× H100:** `--size 416*720 --fps 20 --steam_audio` (streaming) — untested here with the fix.
- **Best quality, 2× GPU:** `--size 480*832 --fps 24` (no steam_audio).
- **Single 4090/5090 (24–32 GB):** `--fp8_kv_cache --block_offload --t5_cpu` — but note `--t5_cpu`
  runs the 11 B umt5-xxl encoder bf16 **on CPU**, which is pathologically slow on weak-CPU hosts
  (stalled 38 min at 0 % GPU on one test) — fine on a fast 128-core host.
