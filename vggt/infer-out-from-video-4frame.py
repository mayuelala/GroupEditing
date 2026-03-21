import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from PIL import Image
import numpy as np
import os
import torchvision.transforms.functional as TF
import cv2


def resolve_hf_model_dir(model_root):
    """Resolve Hugging Face cache root to a snapshot directory with real files."""
    config_file = os.path.join(model_root, "config.json")
    weight_file = os.path.join(model_root, "model.safetensors")
    if os.path.isfile(config_file) and os.path.isfile(weight_file):
        return model_root

    refs_main = os.path.join(model_root, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            revision = f.read().strip()
        candidate = os.path.join(model_root, "snapshots", revision)
        if os.path.isfile(os.path.join(candidate, "config.json")) and os.path.isfile(
            os.path.join(candidate, "model.safetensors")
        ):
            return candidate

    snapshots_root = os.path.join(model_root, "snapshots")
    if os.path.isdir(snapshots_root):
        for snap in sorted(os.listdir(snapshots_root), reverse=True):
            candidate = os.path.join(snapshots_root, snap)
            if os.path.isfile(os.path.join(candidate, "config.json")) and os.path.isfile(
                os.path.join(candidate, "model.safetensors")
            ):
                return candidate

    return None


def load_vggt_model(model_root, device):
    resolved_dir = resolve_hf_model_dir(model_root)
    if resolved_dir is None:
        raise FileNotFoundError(
            "Cannot find VGGT model files under "
            f"{model_root}. Expected config.json and model.safetensors in model root "
            "or under snapshots/<revision>/."
        )
    print(f"Loading VGGT from: {resolved_dir}")
    return VGGT.from_pretrained(resolved_dir).to(device)


def load_and_preprocess_images_from_array(image_array_list, mode="crop"):
    if len(image_array_list) == 0:
        raise ValueError("At least 1 image is required")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.to_tensor
    target_size = 518

    for img_array in image_array_list:
        img = Image.fromarray(img_array)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y:start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)
    if len(image_array_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images


def apply_mask(img_array, mask_array):
    img_array[mask_array < 240] = 255
    return img_array


def read_all_frames(video_path):
    """读取视频的所有帧"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.uint8))

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Video {video_path} contains no frames.")
    return frames


def process_and_save_image(video_path, mask_path, output_folder, model, device, dtype, process_mode="pad"):
    base_name = os.path.basename(video_path).replace('.mp4', '')
    output_file = os.path.join(output_folder, f"{base_name}_aggregated_tokens.npy")
    if os.path.exists(output_file):
        print(f"Existing: for {output_file}")
        return

    img_frames = read_all_frames(video_path)
    mask_frames = read_all_frames(mask_path)

    if len(img_frames) != len(mask_frames):
        print(f"⚠️ Warning: frame count mismatch ({len(img_frames)} vs {len(mask_frames)}) for {video_path}")
        min_len = min(len(img_frames), len(mask_frames))
        img_frames = img_frames[:min_len]
        mask_frames = mask_frames[:min_len]

    masked_imgs = [apply_mask(img, mask) for img, mask in zip(img_frames, mask_frames)]

    # 模型预处理
    images = load_and_preprocess_images_from_array(masked_imgs, mode=process_mode).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # 添加 batch 维度
        aggregated_tokens_list, _ = model.aggregator(images)

    # 保存 token
    np.save(output_file, aggregated_tokens_list[-1].cpu().numpy())
    print(f"✅ Processed and saved: {output_file}")


def process_images_in_folder(folder_path, output_folder, model, device, dtype, process_mode="pad"):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('-origin.mp4'):
            img_path = os.path.join(folder_path, file_name)
            mask_path = os.path.join(folder_path, file_name.replace('-origin.mp4', '-mask.mp4'))

            if not os.path.exists(mask_path):
                print(f"⚠️ Warning: mask not found for {img_path}")
                continue

            process_and_save_image(img_path, mask_path, output_folder, model, device, dtype, process_mode=process_mode)


# ========== 主入口 ==========
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model_root = os.environ.get(
        "VGGT_MODEL_ROOT",
        "./models/facebook/models--facebook--VGGT-1B",
    )
    model = load_vggt_model(model_root, device)
    folder_path = "./test-data/Gemini-out-expand-5"
    output_folder = folder_path + "-vggt"

    process_images_in_folder(folder_path, output_folder, model, device, dtype, process_mode="pad")
