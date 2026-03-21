import os
import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


# 初始化 pipeline
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(path="/cfs-cq/wangxinyu/pretrained_models/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP/diffusion_pytorch_model.safetensors", offload_device="cpu"),
        ModelConfig(path="/cfs-cq/wangxinyu/pretrained_models/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(path="/cfs-cq/wangxinyu/pretrained_models/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(path="/cfs-cq/wangxinyu/pretrained_models/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()


# 输入输出路径
frames_root = "/cfs-cq/wangxinyu/Wan2.1/frames"
result_root = "/cfs-cq/wangxinyu/Wan2.1/result"

# 遍历所有子文件夹
for subdir in os.listdir(frames_root):
    subdir_path = os.path.join(frames_root, subdir)
    if not os.path.isdir(subdir_path):
        continue

    # 创建对应的输出目录
    save_dir = os.path.join(result_root, subdir)
    os.makedirs(save_dir, exist_ok=True)

    # 找到所有 frame 文件并按数字排序
    frame_files = sorted(
        [f for f in os.listdir(subdir_path) if f.startswith("frame-") and f.endswith(".jpg")],
        key=lambda x: int(x.split("-")[1].split(".")[0])
    )

    # 遍历奇数-偶数帧对
    for i in range(0, len(frame_files) - 1, 2):
        frame_odd = os.path.join(subdir_path, frame_files[i])      # 奇数帧
        frame_even = os.path.join(subdir_path, frame_files[i + 1]) # 偶数帧

        idx = int(frame_files[i].split("-")[1].split(".")[0])

        image_first = Image.open(frame_odd)
        image_end = Image.open(frame_even)

        # 判断方向，决定分辨率
        if image_first.height > image_first.width:
            height, width = 832, 480   # 竖屏
        else:
            height, width = 480, 832   # 横屏

        # 生成视频
        video = pipe(
            prompt="",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=image_first,
            end_image=image_end,
            seed=0,
            tiled=True,
            height=height,
            width=width,
        )

        # 保存视频
        save_path = os.path.join(save_dir, f"1.3B-video_{idx}-{idx+1}.mp4")
        save_video(video, save_path, fps=15, quality=5)

        print(f"保存成功: {save_path} ({width}x{height})")
