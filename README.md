<div align="center">
<!-- 左侧：Logo + Group Editing 组合 -->
<p align="center">
  <img src="docs/logo.png" alt="Group Editing Logo" width="92" style="vertical-align:middle;">
  <img src="docs/group-editing-wordmark.svg" alt="Group Editing" width="300" style="vertical-align:middle;">
</p>

<h2><span style="color:#FFFFFF;">Group Editing: Edit Multiple Images in One Go</span></h2>

[Yue Ma](https://mayuelala.github.io/), [Xinyu Wang](https://github.com/cp-cp), [Qianli Ma](https://mqleet.github.io/), [Qinghe Wang](https://qinghew.github.io/), [Mingzhe Zheng](https://scholar.google.com/citations?user=U6bikksAAAAJ&hl=en), [Xiangpeng Yang](https://xiangpengyang.github.io/), [Hao Li](placeholder_url), [Chongbo Zhao](https://github.com/chongbozhao3-coder), [Jixuan Ying](https://hpesojyjx.github.io/), [Harry Yang](https://hyang.org/), [Hongyu Liu](https://kumapowerliu.github.io/), [Qifeng Chen](https://cqf.io/)

<strong>Accepted by CVPR 2026</strong>

<a href='https://arxiv.org/abs/2603.22883'><img src='https://img.shields.io/badge/ArXiv-2603.22883-red'></a>
<a href='https://group-editing.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
[![GitHub](https://img.shields.io/github/stars/mayuelala/GroupEditing?style=social)](https://github.com/mayuelala/GroupEditing)

</div>

## 🎏 Abstract
<b>TL; DR: Group Editing enables consistent editing across multiple images in one go by combining pseudo-video modeling with explicit geometry cues.</b>

<details><summary>CLICK for the full abstract</summary>

> Editing a set of related images with consistent subject identity, style, and structure is challenging due to viewpoint and pose variation. Group Editing reformulates multiple images as a pseudo-temporal sequence and leverages a video-generation prior to improve global consistency. To further enhance cross-view alignment, we integrate VGGT-based geometric correspondence and flow cues into the generation process. Our implementation uses a practical 5-stage pipeline, including mask extraction, input conversion, VGGT token extraction, flow estimation, and final Wan-VACE based generation. This repository provides the research code and engineering pipeline for reproducible group editing experiments.

</details>

## 📀 Demo Video

Please view the demo videos, visual comparisons, and qualitative results on the
[Project Page](https://group-editing.github.io/group-editing/).

## 📋 Changelog

- 2026.03 Initial public release of Group-Editing codebase
- 2026.03 Released Group-Editing LoRA checkpoint `epoch-9.safetensors` on Hugging Face: https://huggingface.co/Heey731/group-editing

## 🚧 Todo

- [ ] Release more demo videos and cases
- [ ] Add one-command pipeline launcher
- [ ] Add config-driven path management (YAML/JSON)
- [ ] Add cleaner benchmark/evaluation scripts
- [ ] Release training details and model cards

## ✨ Features

- **Group-level consistent editing** across multiple input images
- **Pseudo-video reformulation** for improved temporal-like coherence
- **VGGT-based geometry guidance** for better correspondence alignment
- **Mask-aware subject editing** using GroundingDINO + SAM
- **Flow-guided generation** with multi-stage preprocessing
- **Wan-VACE + LoRA integration** for controllable generation

## 🛡 Setup Environment

```bash
# Create conda environment
conda create -n group-edit python=3.10
conda activate group-edit

git clone https://github.com/mayuelala/GroupEditing
# Install dependencies
cd GroupEditing
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU
- Recommended: 24GB+ VRAM (higher VRAM provides smoother inference)

## 📥 Model Download

This project needs several checkpoints from Hugging Face / ModelScope plus your project LoRA.

| Component | Model ID / Source | Local Target Directory | Used In |
|---|---|---|---|
| Grounding DINO | `IDEA-Research/grounding-dino-base` | `./models/IDEA-Research/grounding-dino-base` | `utils/process-origin2mask.py` |
| SAM | `facebook/sam-vit-huge` | `./models/facebook/sam-vit-huge` | `utils/process-origin2mask.py` |
| VGGT | `facebook/VGGT-1B` | `./models/facebook/models--facebook--VGGT-1B` | `vggt/infer-out-from-video-4frame.py` |
| Wan VACE 14B shards | `Wan-AI/Wan2.1-VACE-14B` | `./models/Wan-AI/Wan2.1-VACE-14B` | `infer-test.py` |
| Wan converted T5/VAE | `DiffSynth-Studio/Wan-Series-Converted-Safetensors` | `./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors` | `infer-test.py` |
| Group-Editing LoRA | `Heey731/group-editing` (`epoch-9.safetensors`) | `./models/epoch-9.safetensors` | `infer-test.py` |

### Download core checkpoints (GroundingDINO / SAM / VGGT)

We use pretrained models from GroundingDINO, SAM, and VGGT.  
Our method additionally requires a LoRA checkpoint, provided separately.

```python
from huggingface_hub import snapshot_download

snapshot_download("IDEA-Research/grounding-dino-base", local_dir="./models/grounding-dino-base")
snapshot_download("facebook/sam-vit-huge", local_dir="./models/sam-vit-huge")
snapshot_download("facebook/VGGT-1B", local_dir="./models/VGGT-1B")
```

### Download Wan checkpoints (example with ModelScope)

```bash
# Wan VACE 14B
modelscope download --model Wan-AI/Wan2.1-VACE-14B \
  --local_dir ./models/Wan-AI/Wan2.1-VACE-14B

# Wan converted safetensors (T5 + VAE)
modelscope download --model DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  --local_dir ./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors
```

### Download Group-Editing LoRA checkpoint

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Heey731/group-editing",
    filename="epoch-9.safetensors",
    local_dir="./models"
)
```

> Note: `infer-test.py` currently uses `ckpt_path` at line 45. Please set it to your local LoRA checkpoint path before running.
> The released LoRA file is `./models/epoch-9.safetensors` (source: https://huggingface.co/Heey731/group-editing).

## ⚔️ Group Editing Inference

#### Quick Start (5-stage pipeline)

```bash
# 1) Extract object masks from origin videos
python utils/process-origin2mask.py

# 2) Convert mask/origin videos to pipeline input format
python utils/process-mask2input.py

# 3) Extract VGGT tokens
# Optional: export VGGT_MODEL_ROOT=./models/facebook/models--facebook--VGGT-1B
python vggt/infer-out-from-video-4frame.py

# 4) Compute flow tensors from masks
python utils/2delta-batch-gpu-multi-frame.py

# 5) Run final generation
python infer-test.py
```

#### Input Data Format

- Origin video: `./test-data/Gemini-out/<id>-origin.mp4`
- Object description JSON: `./test-data/gemini-test.json`
- Generated intermediate folders:
  - `./test-data/Gemini-out-expand-5`
  - `./test-data/Gemini-out-expand-5-vggt`
  - `./test-data/Gemini-out-expand-5-map`

## 📁 Project Structure

<details><summary>Click for directory structure</summary>

```text
Group-Editing/
├── diffsynth/                         # Core diffusion framework
│   ├── models/                        # Model definitions (Wan DiT/VACE, encoders, etc.)
│   └── pipelines/                     # Pipeline implementations (wan_video_new.py)
├── utils/
│   ├── process-origin2mask.py         # Stage-1 mask extraction (GroundingDINO + SAM)
│   ├── process-mask2input.py          # Stage-2 input conversion
│   └── 2delta-batch-gpu-multi-frame.py# Stage-4 flow tensor extraction
├── vggt/
│   └── infer-out-from-video-4frame.py # Stage-3 VGGT token extraction
├── infer-test.py                      # Stage-5 final inference
├── models/                            # Local checkpoints (ignored by git)
├── test-data/                         # Local data and intermediate files (optional)
├── requirements.txt
└── README.md
```

</details>

## 🔧 Key Modifications

This repository is built on top of DiffSynth-Studio and includes project-specific edits for Group Editing:

### 1. `infer-test.py`

- Integrates LoRA loading for Wan-VACE pipeline
- Loads and injects VGGT tokens (`vggt_tensor`) and flow tensors (`flow_tensor`)
- Implements practical task loop for grouped editing generation

### 2. `vggt/infer-out-from-video-4frame.py`

- Adds masked-frame token extraction for video-style group inputs
- Supports Hugging Face cache-style model root resolution (`snapshots/<revision>`)

### 3. `utils/process-origin2mask.py` + `utils/2delta-batch-gpu-multi-frame.py`

- Stage-1 object mask extraction with GroundingDINO and SAM
- Stage-4 contour/TPS-based flow map generation for guidance

### 4. `diffsynth/models/stepvideo_text_encoder.py`

- Added import fallback for transformers API compatibility across versions:
  - `from transformers import PretrainedConfig, PreTrainedModel`
  - fallback to `configuration_utils` / `modeling_utils`

## 📍 Citation

If you use this code, please cite:

```bibtex
@article{groupediting2026,
  title={Group Editing: Edit Multiple Images in One Go},
  author={Ma, Yue and Wang, Xinyu and Ma, Qianli and Wang, Qinghe and Zheng, Mingzhe and Yang, Xiangpeng and Li, Hao and Zhao, Chongbo and Ying, Jixuan and Liu, Hongyu and Chen, Qifeng},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## 📜 License

This project is released under the Apache-2.0 License.
See [LICENSE](LICENSE) for details.

## 💗 Acknowledgements

This repository builds upon:

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [VGGT](https://github.com/facebookresearch/vggt)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)

Thanks to the original authors and communities for open-sourcing their work.

## 🧿 Maintenance

This repository is maintained for research and reproducibility.
If you find issues or have suggestions, please open an issue or discussion thread.
