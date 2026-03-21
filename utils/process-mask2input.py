import cv2
import numpy as np
import os
import shutil

def fill_mask_holes(mask_binary: np.ndarray) -> np.ndarray:
    """填充 mask 内部空洞，返回填充后的二值掩码。"""
    mask_uint8 = (mask_binary * 255).astype(np.uint8)
    h, w = mask_uint8.shape

    # 外部加边框，防止外部背景被误识为空洞
    padded = cv2.copyMakeBorder(mask_uint8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_filled = padded.copy()
    cv2.floodFill(flood_filled, None, (0, 0), 255)

    # 取反 -> 得到内部空洞 -> 与原mask合并
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    filled_mask = cv2.bitwise_or(padded, flood_filled_inv)
    filled_mask = filled_mask[1:h+1, 1:w+1]

    return (filled_mask > 127).astype(np.uint8)


def apply_mask_and_gray(origin_path, mask_path, output_path, mask_save_path, expand_pixels=5):
    """
    扩展 mask、填充空洞，并用于灰色化，同时保存新的 mask 视频。
    """
    cap_origin = cv2.VideoCapture(origin_path)
    cap_mask = cv2.VideoCapture(mask_path)
    if not cap_origin.isOpened() or not cap_mask.isOpened():
        raise RuntimeError(f"❌ 无法打开视频: {origin_path} 或 {mask_path}")

    fps = cap_origin.get(cv2.CAP_PROP_FPS)
    width = int(cap_origin.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_origin.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_gray = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    out_mask = cv2.VideoWriter(mask_save_path, fourcc, fps, (width, height), isColor=False)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1))
    print(f"开始处理 {os.path.basename(origin_path)} ...")

    while True:
        ret_origin, frame_origin = cap_origin.read()
        ret_mask, frame_mask = cap_mask.read()
        if not (ret_origin and ret_mask):
            break

        # mask -> 灰度 -> 二值化 -> 填充空洞 -> 膨胀
        mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        mask_binary = (mask_gray > 127).astype(np.uint8)
        # mask_binary = fill_mask_holes(mask_binary)
        if expand_pixels > 0:
            mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)

        # 保存新 mask
        out_mask.write(mask_binary * 255)

        # 应用灰色化
        mask_3ch = np.stack([mask_binary] * 3, axis=-1)
        gray_color = np.full_like(frame_origin, 127, dtype=np.uint8)
        frame_result = np.where(mask_3ch == 1, gray_color, frame_origin)
        out_gray.write(frame_result)

    cap_origin.release()
    cap_mask.release()
    out_gray.release()
    out_mask.release()
    print(f"✅ 完成: {os.path.basename(output_path)} (含新mask)")


def process_folder(input_folder, output_folder, expand_pixels=5):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if not filename.endswith("-origin.mp4"):
            continue

        basename = filename.replace("-origin.mp4", "")
        origin_path = os.path.join(input_folder, f"{basename}-origin.mp4")
        mask_path = os.path.join(input_folder, f"{basename}-mask.mp4")
        output_video_path = os.path.join(output_folder, f"{basename}.mp4")
        output_origin_path = os.path.join(output_folder, f"{basename}-origin.mp4")
        output_mask_filled_path = os.path.join(output_folder, f"{basename}-mask.mp4")  # 仅保存新mask

        if not os.path.exists(mask_path):
            print(f"⚠️ 跳过：未找到 mask 文件 {mask_path}")
            continue

        try:
            shutil.copy2(origin_path, output_origin_path)
            apply_mask_and_gray(origin_path, mask_path, output_video_path, output_mask_filled_path, expand_pixels)
        except Exception as e:
            print(f"❌ 处理 {basename} 时出错: {e}")


if __name__ == "__main__":
    expand_pixels = 5
    input_folder = "./test-data/Gemini-out"
    output_folder = input_folder + f"-expand-{expand_pixels}"
    process_folder(input_folder, output_folder, expand_pixels=expand_pixels)
