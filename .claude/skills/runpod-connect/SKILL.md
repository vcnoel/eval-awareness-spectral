---
name: runpod-connect
description: Connect to a RunPod GPU pod over SSH and run this repo's scaling experiments on it (Qwen ladder crystallization + spectral transfer). Use when the user wants to run the eval-awareness experiments on a rented cloud GPU (A100/H100), set up the pod environment, or drive it from a local machine. Covers the exact gotchas (PEP 668, torch-safe install, HF_HOME, disk) that bite on a fresh pod.
---

# Connect to RunPod and run the scaling experiments

The local machine's shell tools do NOT run on the pod — drive the pod over SSH from the
local shell. RunPod exposes the pod's SSH on a **Direct TCP** endpoint (not port 22 on the
pod's public IP): find it in the pod page under *Direct TCP ports*, e.g. `195.26.233.38:58513 -> :22`.

## 1. Authorize your key on the pod (instant, no pod restart)

The pod must trust the **local** private key your SSH will use (default `~/.ssh/id_ed25519`).
Do NOT run RunPod's `ssh-keygen` wizard if you already have a key — it makes a *different* key.

Print your local public key:
```bash
cat ~/.ssh/id_ed25519.pub    # if missing: ssh-keygen -t ed25519
```
Open the pod's **web terminal** (button on the pod page) and paste (with YOUR pubkey):
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "<YOUR_PUBKEY_LINE>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo KEY_ADDED
```
(The RunPod "SSH public key" UI field also works but usually needs a pod restart; the
web-terminal line is instant.)

Test from the local shell (substitute the pod's host/port):
```bash
POD="-p 58513 root@195.26.233.38"
ssh $POD -o StrictHostKeyChecking=accept-new 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /workspace | tail -1'
```

## 2. Set up the environment (fresh pod gotchas)

RunPod PyTorch images have CUDA torch preinstalled but a **PEP 668 externally-managed**
Python, and are missing transformers/datasets/etc. Install torch-safely:
```bash
ssh $POD 'cd /workspace && git clone -q https://github.com/vcnoel/eval-awareness-spectral.git && cd eval-awareness-spectral \
  && pip install -q --break-system-packages --no-deps "spectral-trust==0.2.2" \
  && pip install -q --break-system-packages "transformers>=4.44" accelerate scipy scikit-learn datasets pandas tqdm matplotlib seaborn'
```
Key points:
- `--break-system-packages` — required (PEP 668) to install into the same env as the pod's torch.
- `--no-deps` on spectral-trust — do NOT let it reinstall torch (would clobber the CUDA build with a CPU wheel). Then install its real deps explicitly (matplotlib + seaborn are needed or import fails).
- `spectral_trust.__version__` may be absent — harmless; the repo falls back to "0.2.2".

## 3. Disk: point HF at the volume, container disk stays tiny

Container disk (`/`, ~30 GB, also holds `/tmp`) is small; the persistent volume `/workspace`
is huge. Set `HF_HOME=/workspace/hf` so ~120 GB of weights land on the volume — then container
disk never fills and you do NOT need to increase it. Verify: `df -h / ; df -h /workspace`.

## 4. Run the scaling ladder (detached, survives disconnect)

**Detach gotcha (important):** a plain `nohup python … & echo started` inside a *chained*
one-line SSH command gets SIGHUP-killed when the SSH session closes (its process group dies).
Use `setsid` + `</dev/null` + `disown` so it fully detaches:
```bash
ssh $POD 'cd /workspace/eval-awareness-spectral && export PYTHONPATH=src HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 \
  && setsid nohup python scripts/run_pod_scaling.py --max-b 32 >scaling.log 2>&1 </dev/null & disown'
# verify it actually survived (run in a SEPARATE ssh a few seconds later):
ssh $POD 'pgrep -af run_pod_scaling | grep -v grep; tail -3 /workspace/eval-awareness-spectral/scaling.log'
```
- Smoke first: `--max-b 3` (Qwen 0.5/1.5/3B, ~10 GB) to confirm the pipeline, then `--max-b 32`.
- VRAM: bf16 fits through 32B on an 80 GB A100; 72B needs 2 GPUs (omitted). fp32 buys nothing —
  diagnostics are float32/float64 regardless of weight dtype.
- The driver downloads each Qwen2.5 model, extracts position-matched task-subgraph spectral +
  per-token activations, then prints the CRYSTALLIZATION, TRANSFER, and per-model SELECTIVITY
  tables (the F1–F3 decision, see `docs/laws.md`). CSVs land in `results/pod_scaling/`.

## 4b. Monitor a long run from the local machine (one notification when done)

SSH commands' stdout is buffered until the session returns, so don't pipe the run through
`grep|tail` in the foreground (you see nothing until it ends). Instead poll the log file in a
loop that exits when a near-end marker appears or the process dies:
```bash
D=/workspace/eval-awareness-spectral
until ssh $POD "grep -q 'SPECTRAL TRANSFER' $D/scaling.log 2>/dev/null || ! pgrep -f run_pod_scaling >/dev/null"; do sleep 120; done
ssh $POD "grep -E '=====|C4_taskindep|C10_eval_rank|mean OFF|params_b|qwen2.5|FAILED|OutOfMemory' $D/scaling.log | grep -viE 'it/s' | tail -60"
```
The `until` covers both success (marker present) and failure (process gone), so it never hangs
silently on a crash. Progress spot-check any time: `ssh $POD "nvidia-smi --query-gpu=memory.used --format=csv,noheader; ls $D/results/pod_scaling/*/diagnostics_long.csv | wc -l"`.

## Pushing code updates to the pod (do NOT rely on `git pull`)

The pod often can't `git pull`: if the repo is **private** (or toggled private after the
initial clone), `git fetch` errors `could not read Username` and `raw.githubusercontent`
404s — both need auth the pod doesn't have. **Push updated files with `scp` from the local
machine instead** (you already have SSH):
```bash
scp -P <PORT> src/eval_awareness_spectral/foo.py root@<HOST>:/workspace/eval-awareness-spectral/src/eval_awareness_spectral/
scp -P <PORT> scripts/run_pod_scaling.py root@<HOST>:/workspace/eval-awareness-spectral/scripts/
ssh $POD 'cd /workspace/eval-awareness-spectral && python -c "import ast; ast.parse(open(\"scripts/run_pod_scaling.py\").read()); print(\"SYNTAX OK\")"'
```
scp bypasses GitHub entirely and always works over the existing SSH channel.

## The network-volume quota (the real disk limit) — and how to grow it

Two different disks: **container disk** (`/`, ~30 GB, holds pip packages + `/tmp`, resets on
pod edit) and the **network volume** (`/workspace`, a *provisioned quota* you pay for). `df`
shows the underlying cluster (100s of TB) but writes fail with **`Disk quota exceeded (os error
122)`** at the provisioned quota. That is the limit that stops big model downloads.

- To grow it: RunPod **"Manage network storage"** → increase the volume GB. Do **NOT** use
  **"Edit Pod"** — that resets the *container* (you lose all pip installs) and does not change
  the volume anyway. Growing the network volume may still restart the pod (container resets,
  `/workspace` persists → results safe, but reinstall deps per step 2).
- To avoid growing it: run one model at a time with **`--purge-weights`** (deletes each model's
  weights after its features are extracted, so peak usage ≈ one model). On a ~30 GB volume this
  safely covers **≤9B** models (a 14B ≈ 28 GB is too tight even alone). `run_pod_scaling.py`
  is **resumable** (skips rungs whose `token_acts.pkl` exists) and aggregates over ALL present
  rungs, so you can wipe model weights between stages without losing results.

## Strategy: breadth over depth on a single mid GPU

For scaling laws, prefer **many families ≤14B** over one family to 72B: multiple within-family
ladders (Qwen2.5 0.5–14B, SmolLM2 0.13–1.7B, Phi, Gemma-2, Llama-3.x) kill the "family artifact"
objection and enrich the cross-architecture transfer matrix. `run_pod_scaling.py`'s `LADDER`
holds them; **open** families (Qwen/SmolLM2/Phi/Mistral) need no token, **gated** ones
(Gemma/Llama) auto-skip unless `HF_TOKEN` is exported and the licenses are accepted.

## Environment facts learned (RunPod PyTorch A100 pod)
- Base image ships CUDA **torch** but **not** transformers/datasets/scikit-learn/matplotlib/seaborn/accelerate.
- Python is PEP 668 externally-managed → every `pip install` needs `--break-system-packages`.
- User is `root`; home is `/root`; persistent volume is `/workspace` (huge, network MFS);
  container disk `/` is ~30 GB and also backs `/tmp` — keep big writes off it via `HF_HOME`.
- Direct-TCP SSH endpoint (host\:port) changes every time you recreate a pod; the authorized_keys
  entry is per-pod (re-add the key after recreating, or set it in RunPod account settings).
- `spectral_trust.__version__` may be missing on 0.2.2 — harmless (repo falls back).

## 5. Pull results back / interpret

Copy the printed tables from `scaling.log`, or `scp` the CSVs:
```bash
scp -P 58513 -r root@195.26.233.38:/workspace/eval-awareness-spectral/results/pod_scaling ./results/
```
Then locally: `python -m eval_awareness_spectral.law_mining` etc. re-print from the CSVs.

## Quick reference — the whole thing in one block
```bash
POD="-p <PORT> root@<HOST>"                                  # from RunPod Direct TCP
cat ~/.ssh/id_ed25519.pub                                    # paste into pod web terminal authorized_keys
ssh $POD -o StrictHostKeyChecking=accept-new 'nvidia-smi -L'
ssh $POD 'cd /workspace && git clone -q https://github.com/vcnoel/eval-awareness-spectral.git && cd eval-awareness-spectral && pip install -q --break-system-packages --no-deps spectral-trust==0.2.2 && pip install -q --break-system-packages "transformers>=4.44" accelerate scipy scikit-learn datasets pandas tqdm matplotlib seaborn'
ssh $POD 'cd /workspace/eval-awareness-spectral && export PYTHONPATH=src HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 && nohup python scripts/run_pod_scaling.py --max-b 32 > scaling.log 2>&1 & echo started'
```
