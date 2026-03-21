import os
import torch
import pandas as pd
import numpy as np
import cv2
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import random

def auto_fix_vace_keys(model, state_dict):
    """自动检测并修复 vace 模型键名的差异（.module. / 无 .module.）"""
    model_keys = list(model.state_dict().keys())
    has_module = any(".module." in k for k in model_keys)
    fixed_state_dict = {}

    for k, v in state_dict.items():
        if not k.startswith("vace."):
            continue
        new_k = k.replace("vace.", "", 1)

        if has_module and ".module." not in new_k:
            new_k = new_k.replace(".norm_q.", ".norm_q.module.")
            new_k = new_k.replace(".norm_k.", ".norm_k.module.")
            new_k = new_k.replace(".cross_attn.", ".cross_attn.module.")
            new_k = new_k.replace(".norm3.", ".norm3.module.")
            new_k = new_k.replace(".vace_patch_embedding.", ".vace_patch_embedding.module.")
        elif not has_module and ".module." in new_k:
            new_k = new_k.replace(".module.", ".")
        fixed_state_dict[new_k] = v

    return fixed_state_dict


def align_to_16(x: int) -> int:
    """向下取整到16的倍数"""
    return (x // 16) * 16


# =============================
# 1. 参数配置
# =============================
video_base_path = "./test-data/Gemini-out-expand-5"
vggt_base_path = f"{video_base_path}-vggt"
flow_base_path = f"{video_base_path}-map"
ckpt_path = "./models/epoch-9.safetensors"
print(f"Loading checkpoint from {ckpt_path}")
out_base_path = f"./test-out"

# =============================
# 2. prompt / base_name 列表
# =============================

tasks = [
    ("robotic fox with segmented armor plates and glowing tail core", "351"),
]

random.shuffle(tasks)
# =============================
# 3. 初始化 pipeline
# =============================
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[
            "./models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00001-of-00007.safetensors",
            "./models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00002-of-00007.safetensors",
            "./models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00003-of-00007.safetensors",
            "./models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00004-of-00007.safetensors",
            "./models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00005-of-00007.safetensors",
            "./models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00006-of-00007.safetensors",
            "./models/Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model-00007-of-00007.safetensors",
        ]),
        ModelConfig(path="./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"),
        ModelConfig(path="./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"),
    ],
)

# =============================
# 加载 checkpoint
# =============================
state_dict = load_state_dict(ckpt_path)

fixed_vace_state_dict = auto_fix_vace_keys(pipe.vace, state_dict)
missing_vace, unexpected_vace = pipe.vace.load_state_dict(fixed_vace_state_dict, strict=False)
print(f"[LoRA Loding] vace loaded, missing: {missing_vace}, unexpected: {unexpected_vace}")

if hasattr(pipe, "dit"):
    pipe.load_lora(pipe, ckpt_path, alpha=1.0)
    print(f"[LoRA Loding] dit LoRA weights loaded")
else:
    print(f"[LoRA Loding] WARNING: pipe.dit 没有 load_lora_weights 方法，跳过 dit LoRA 加载")

pipe.enable_vram_management()


# =============================
# 4. 先遍历 seed，再遍历 prompt
# =============================
for seed in range(1):
    print(f"\n================== Seed {seed} ==================\n")

    for obj, base_name in tasks:
        prompt = f"same {obj} in four scenes."
        print(f"\n▶️ [Seed {seed}] 推理任务: {base_name} ({prompt})")

        video_path = os.path.join(video_base_path, f"{base_name}.mp4")
        mask_path = os.path.join(video_base_path, f"{base_name}-mask.mp4")
        vggt_tensor_path = os.path.join(vggt_base_path, f"{base_name}-origin_aggregated_tokens.npy")
        flow_tensor_path = os.path.join(flow_base_path, f"{base_name}-map.npy")
        os.makedirs(out_base_path, exist_ok=True)
        target_video = os.path.join(out_base_path, f"{base_name}-seed-{seed}-{prompt.replace(' ', '_')[:30]}-output.mp4")
        if os.path.exists(target_video):
            print(f"{target_video} already exists, skipping...")
            continue
        if not all(os.path.exists(p) for p in [video_path, mask_path, vggt_tensor_path, flow_tensor_path]):
            print(f"⚠️ 缺少必要文件，跳过 {base_name}")
            continue

        # 自动读取视频尺寸并对齐到16倍数
        cap = cv2.VideoCapture(video_path)
        width_raw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_raw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        width, height = align_to_16(width_raw), align_to_16(height_raw)

        # width, height = 512, 512
        print(f"📏 原始尺寸: {width_raw}x{height_raw} → 对齐后: {width}x{height}")

        vggt_tensor = torch.from_numpy(np.load(vggt_tensor_path)).to(torch.bfloat16).to("cuda")
        flow_tensor = torch.from_numpy(np.load(flow_tensor_path)).to(torch.float32).to("cuda")

        frame_num = 4
        video = VideoData(video_path, height=height, width=width)
        video = [video[i] for i in range(frame_num)]

        video_mask = VideoData(mask_path, height=height, width=width)
        video_mask = [video_mask[i] for i in range(frame_num)]


        video_out = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            vace_video=video,
            vace_video_mask=video_mask,
            num_frames=frame_num,
            height=height, width=width,
            seed=seed, tiled=True,
            vggt_tensor=vggt_tensor,
            flow_tensor=flow_tensor,
        )

        save_video(video_out, target_video, fps=1, quality=5)
        print(f"✅ [Seed {seed}] Saved: {target_video}\n")
