# HPC Deployment Guide (NYU Torch)

This guide describes how to deploy and run the Foreign Whispers pipeline
on an HPC cluster without Docker. It was developed and tested on
**NYU Torch HPC** using:

- Login node: `login.torch.hpc.nyu.edu`
- GPU compute node: 1× NVIDIA H100 (80 GB), `h100_tandon` partition
- Job scheduler: Slurm
- User-space container runtime: Apptainer / Singularity
- Python toolchain: `uv`
- System modules: `anaconda3/2025.06` (used only to provision `ffmpeg`
  into `/scratch`)

The same pattern applies to any Slurm-managed cluster with Apptainer and
an NVIDIA GPU with ≥ 16 GB of VRAM. Substitute cluster-specific
module names and partition names as appropriate.

> **Why not Docker on HPC?** HPC clusters do not expose the Docker
> daemon because it requires root. Apptainer (formerly Singularity)
> provides a rootless alternative that can consume the same OCI images.

---

## Overview

You will end up with four long-running services, all reachable on
`localhost` of the compute node:

| Port  | Service                                 | How it runs                                  |
|-------|-----------------------------------------|----------------------------------------------|
| 8000  | Whisper STT (`speaches-ai/speaches`)    | Apptainer `.sif`                             |
| 8020  | Chatterbox TTS (`travisvn/chatterbox`)  | Apptainer `.sif`                             |
| 8080  | FastAPI orchestrator                    | `uv run uvicorn api.src.main:app …`          |
| 8888  | Jupyter Lab (optional, for dev)         | `uv run jupyter lab …`                       |

From your laptop you reach them via SSH local-port forwarding.

```
┌──────────────────┐           ┌───────────────────────────────┐
│ Laptop (PS/Shell)│           │ Torch login node              │
│                  │  ssh -L   │                               │
│ localhost:8080 ──┼──────────▶│       ssh → GPU node (gh0XX)  │
│ localhost:8888 ──┼──────────▶│        ├─ Whisper  :8000      │
│                  │           │        ├─ Chatterbox :8020    │
│ (browser, curl,  │           │        ├─ FastAPI   :8080     │
│  Python scripts) │           │        └─ Jupyter   :8888     │
└──────────────────┘           └───────────────────────────────┘
```

---

## 1. First-time cluster setup

These steps run once per cluster account.

### 1.1 Log in

```bash
ssh <netid>@login.torch.hpc.nyu.edu
# Complete NYU MFA (Duo / Microsoft PIN) when prompted.
```

### 1.2 Clone the repository into `/scratch`

`/home` on Torch is small and read-only on compute nodes for package
installs. Use `/scratch/$USER` for all project work.

```bash
mkdir -p /scratch/$USER/repos
cd /scratch/$USER/repos
git clone https://github.com/yuelihe2-svg/foreign-whispers-project.git foreign-whispers
cd foreign-whispers
```

### 1.3 Install `uv` into your user environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env         # or re-login
uv --version
```

### 1.4 Sync Python dependencies

```bash
cd /scratch/$USER/repos/foreign-whispers
uv sync
```

This creates `/scratch/$USER/repos/foreign-whispers/.venv/` with
PyTorch, faster-whisper, argostranslate, FastAPI, and friends.

### 1.5 Install `ffmpeg` into user space

`ffmpeg` is not on Torch's default `$PATH`, and the API container image
is not available. The simplest reproducible install is through
Miniforge / Anaconda into `/scratch`:

```bash
module load anaconda3/2025.06
conda create -y --prefix /scratch/$USER/conda-envs/fw-tools \
    --override-channels -c conda-forge ffmpeg
```

The `--override-channels -c conda-forge` flags bypass Anaconda's
default-channel Terms-of-Service prompt that some fresh accounts will
otherwise hit on `conda create`.

### 1.6 Pull the service containers (Apptainer)

```bash
mkdir -p /scratch/$USER/sif
cd /scratch/$USER/sif

# STT — speaches (Whisper faster-whisper backend)
apptainer pull speaches.sif \
    docker://ghcr.io/speaches-ai/speaches:latest-cuda-12.6.3

# TTS — Chatterbox
apptainer pull chatterbox.sif \
    docker://travisvn/chatterbox-tts-api:latest
```

These can be 5–10 GB each and take a few minutes.

### 1.7 (Optional) Upload YouTube cookies

If YouTube challenges downloads from Torch with
*"Sign in to confirm you're not a bot"*, export cookies from a
logged-in browser on your laptop, then from your laptop:

```powershell
# Windows PowerShell
scp cookies.txt <netid>@login.torch.hpc.nyu.edu:/scratch/<netid>/cookies.txt
```

```bash
# macOS / Linux
scp cookies.txt <netid>@login.torch.hpc.nyu.edu:/scratch/$USER/cookies.txt
```

Cookies must be in **Netscape** format. The first line should read
`# Netscape HTTP Cookie File`.

---

## 2. Launching a session

These steps run every time you want to use the cluster.

### 2.1 Allocate a GPU node

From the login node:

```bash
srun --partition=h100_tandon \
     --gres=gpu:h100:1 \
     --cpus-per-task=8 \
     --mem=64G \
     --time=04:00:00 \
     --pty /bin/bash
```

Slurm will assign you a node (e.g. `gh004`). Take note of its hostname
— you will need it from your laptop for the SSH tunnel.

> **GPU-utilization policy**: Torch periodically audits allocated GPUs
> and cancels jobs whose GPUs idle for too long. Keep the pipeline
> actively working, or end the session when you are done.

### 2.2 Start the STT container

In the GPU-node shell:

```bash
mkdir -p /scratch/$USER/fw-logs

apptainer run --nv \
    --env 'WHISPER__MODEL=Systran/faster-whisper-medium' \
    --bind /scratch/$USER/hf-cache:/home/ubuntu/.cache/huggingface/hub \
    /scratch/$USER/sif/speaches.sif \
    > /scratch/$USER/fw-logs/whisper.log 2>&1 &

# Verify
sleep 20
curl http://localhost:8000/health
```

### 2.3 Start the TTS container

```bash
apptainer run --nv \
    --env 'DEVICE=cuda' \
    --env 'DEFAULT_MODEL=multilingual' \
    --env 'PORT=8020' \
    --bind /scratch/$USER/chatterbox-models:/app/models \
    --bind /scratch/$USER/repos/foreign-whispers/pipeline_data/speakers:/app/voices \
    /scratch/$USER/sif/chatterbox.sif \
    > /scratch/$USER/fw-logs/chatterbox.log 2>&1 &

# Verify
sleep 30
curl http://localhost:8020/v1/models
```

### 2.4 Start the FastAPI orchestrator

```bash
cd /scratch/$USER/repos/foreign-whispers

# Put user-space ffmpeg on PATH for this shell
export PATH=/scratch/$USER/conda-envs/fw-tools/bin:$PATH
which ffmpeg      # should print /scratch/<netid>/conda-envs/fw-tools/bin/ffmpeg

# Wire up the service URLs (note: host-networking, so localhost works)
export CHATTERBOX_API_URL=http://localhost:8020
export FW_STT_BASE_URL=http://localhost:8000
export FW_TTS_BASE_URL=http://localhost:8020
export FW_WHISPER_MODEL=base       # or medium; must match STT container

# Optional: cookies for YouTube downloads
export YT_COOKIES_FILE=/scratch/$USER/cookies.txt

# Optional: Hugging Face token for pyannote diarization
# export FW_HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx

nohup uv run uvicorn api.src.main:app \
    --host 0.0.0.0 --port 8080 \
    > /scratch/$USER/fw-logs/api.log 2>&1 &

# Verify (can take 30–60 s on first start while uv resolves deps)
for i in 1 2 3 4 5 6; do
    sleep 10
    if curl -sf http://localhost:8080/healthz; then break; fi
done
```

### 2.5 (Optional) Start Jupyter Lab

```bash
cd /scratch/$USER/repos/foreign-whispers
nohup uv run jupyter lab \
    --no-browser --ip=0.0.0.0 --port=8888 \
    --ServerApp.token='' --ServerApp.password='' \
    > /scratch/$USER/fw-logs/jupyter.log 2>&1 &
```

> The empty token is only safe because the port is never exposed outside
> the compute node — you reach it exclusively through the SSH tunnel
> below.

---

## 3. Reaching the cluster from your laptop

From a laptop terminal (leave this window open — closing it breaks the
tunnel):

```bash
ssh -N \
    -L 8080:localhost:8080 \
    -L 8000:localhost:8000 \
    -L 8020:localhost:8020 \
    -L 8888:localhost:8888 \
    -J <netid>@login.torch.hpc.nyu.edu \
    <netid>@<gpu-node-hostname>
```

Replace `<gpu-node-hostname>` with the node Slurm assigned in step 2.1
(e.g. `gh004.torch.hpc.nyu.edu`). Complete MFA when prompted. The `-J`
flag uses the login node as a jump host, because GPU nodes are not
reachable directly from the public internet.

Now on your laptop:

- FastAPI:  http://localhost:8080/healthz
- Jupyter:  http://localhost:8888/lab
- STT:      http://localhost:8000/health
- TTS:      http://localhost:8020/v1/models

---

## 4. Running the pipeline

Option A — via the notebook:

1. Open Jupyter Lab in the browser at http://localhost:8888/lab.
2. Navigate to `notebooks/pipeline_end_to_end/pipeline_end_to_end.ipynb`.
3. Run all cells.

Option B — via Python SDK from your laptop:

```python
# Must be running on the laptop, with the SSH tunnel active.
from foreign_whispers import FWClient

fw = FWClient(base_url="http://localhost:8080")

video_id = fw.download("https://www.youtube.com/watch?v=GYQ5yGV_-Oc")
fw.transcribe(video_id)
fw.translate(video_id, target_lang="es")
fw.tts(video_id, alignment=False)
fw.stitch(video_id)
```

Option C — with `curl` from the GPU node:

```bash
curl -X POST http://localhost:8080/api/download \
    -H 'Content-Type: application/json' \
    -d '{"url": "https://www.youtube.com/watch?v=GYQ5yGV_-Oc"}'
```

Artifacts land under
`/scratch/$USER/repos/foreign-whispers/pipeline_data/api/`. Copy the
final dubbed video to your laptop with `scp`:

```bash
scp '<netid>@login.torch.hpc.nyu.edu:/scratch/<netid>/repos/foreign-whispers/pipeline_data/api/dubbed_videos/*/*.mp4' .
```

---

## 5. Shutting down

Inside the GPU-node shell:

```bash
# Stop the API
pkill -f 'uvicorn api.src.main:app'

# Stop Jupyter
pkill -f 'jupyter lab'

# Stop the containers (if still running)
pkill -f 'apptainer run .*speaches.sif'
pkill -f 'apptainer run .*chatterbox.sif'

# Release the GPU node
exit
```

Close the laptop-side SSH tunnel window (`Ctrl+C`).

---

## Troubleshooting

### `curl http://localhost:8080/healthz` returns *Connection refused*

The API takes 30–60 seconds to come up on the first `uv run` because
`uv` may still be resolving dependencies. Retry every 10 s for up to a
minute. Also confirm the process is alive:

```bash
pgrep -af 'uvicorn api.src.main:app'
tail -n 100 /scratch/$USER/fw-logs/api.log
```

### Downloads fail with *Sign in to confirm you're not a bot*

YouTube has flagged the Torch IP range. Upload fresh `cookies.txt`
(see §1.7), set `YT_COOKIES_FILE`, and restart the API (§2.4).

### `ffmpeg: command not found` during stitch

`PATH` was not exported in the shell where you launched uvicorn.
`pkill` the API, re-`export PATH=/scratch/$USER/conda-envs/fw-tools/bin:$PATH`,
and re-run the `nohup uv run uvicorn …` command.

### `RuntimeError: Failed to execute rubberband` during TTS

Call TTS with `alignment=False`. This fork's `tts_engine.py` skips
`pyrubberband` entirely in that mode; the alignment is still performed
via segment-level pad/trim. Installing `rubberband-cli` into
`/scratch/$USER/conda-envs/fw-tools` via `conda-forge` is not
straightforward; the code fix is the pragmatic route.

### GPU idle watchdog cancels my Slurm job

Torch cancels jobs whose GPU utilization stays near zero. Either keep
the pipeline busy or batch your work into a single session. You can
check utilization with `nvidia-smi` on the GPU node.

### Apptainer pull fails with *unauthorized* / rate-limited

Log in to Docker Hub from the node first:

```bash
apptainer remote login --username <dockerhub-user> docker://docker.io
```

Or pull the image on a machine you control and transfer the `.sif`
to `/scratch/$USER/sif/`.
