import os
import cv2
import numpy as np
import torch
import decord
from tqdm import tqdm


def read_video_frames(video_path):
    """读取多帧 mask 视频，返回 torch.Tensor [T, H, W]，值范围[0,1]"""
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    T = len(vr)
    frames = vr.get_batch(range(T)).asnumpy()  # [T, H, W, 3]
    frames = frames.mean(axis=-1) / 255.0  # 灰度化
    frames = (frames > 0.9).astype(np.float32)
    return torch.from_numpy(frames).float()  # 转为 torch.Tensor


def extract_contour(mask_np):
    """提取最大连通域轮廓坐标"""
    mask_bin = (mask_np * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2), dtype=np.float32)
    cnt = max(cnts, key=cv2.contourArea)
    return cnt.reshape(-1, 2).astype(np.float32)


def normalize_contour(contour):
    """平移 + 旋转归一化"""
    if len(contour) == 0:
        return contour, np.zeros(2), np.eye(2)
    center = contour.mean(axis=0)
    pts = contour - center
    _, _, vh = np.linalg.svd(pts)
    pts = pts @ vh.T
    return pts, center, vh


def compute_correspondence(mask1, mask2, device="cuda"):
    """计算 mask1 -> mask2 的像素位移场 (GPU TPS 版)，返回 np.array [2, H, W]"""
    H, W = mask1.shape
    cnt1 = extract_contour(mask1.cpu().numpy())
    cnt2 = extract_contour(mask2.cpu().numpy())
    if len(cnt1) == 0 or len(cnt2) == 0:
        return np.zeros((2, H, W), dtype=np.float32)

    # ===== 归一化 =====
    pts1, c1, R1 = normalize_contour(cnt1)
    pts2, c2, R2 = normalize_contour(cnt2)

    # ===== 转 GPU =====
    pts1 = torch.tensor(pts1, dtype=torch.float32, device=device)
    pts2 = torch.tensor(pts2, dtype=torch.float32, device=device)
    c1 = torch.tensor(c1, dtype=torch.float32, device=device)
    c2 = torch.tensor(c2, dtype=torch.float32, device=device)
    R1 = torch.tensor(R1, dtype=torch.float32, device=device)
    R2 = torch.tensor(R2, dtype=torch.float32, device=device)

    # ===== 最近邻匹配 =====
    diff = pts1.unsqueeze(1) - pts2.unsqueeze(0)  # [N1, N2, 2]
    dist = torch.sum(diff ** 2, dim=-1)
    idxs = torch.argmin(dist, dim=1)

    pts1_world = pts1 @ R1 + c1
    pts2_world = pts2[idxs] @ R2 + c2

    # ===== Thin-Plate Spline =====
    def tps_kernel(x, y):
        dist2 = torch.cdist(x, y, p=2) ** 2
        return dist2 * torch.log(dist2 + 1e-6)

    K = tps_kernel(pts1_world, pts1_world)
    P = torch.cat([torch.ones_like(pts1_world[:, :1]), pts1_world], dim=1)
    L = torch.cat([
        torch.cat([K, P], dim=1),
        torch.cat([P.T, torch.zeros((3, 3), device=device)], dim=1)
    ], dim=0)

    Vx = torch.cat([pts2_world[:, 0], torch.zeros(3, device=device)])
    Vy = torch.cat([pts2_world[:, 1], torch.zeros(3, device=device)])

    λ = 1e-3
    I = torch.eye(L.shape[0], device=device)
    L_reg = L + λ * I

    try:
        W_x = torch.linalg.solve(L_reg, Vx)
        W_y = torch.linalg.solve(L_reg, Vy)
    except torch._C._LinAlgError:
        L_pinv = torch.linalg.pinv(L_reg)
        W_x = L_pinv @ Vx
        W_y = L_pinv @ Vy

    # ===== 前景点插值 =====
    ys, xs = torch.where(mask1 > 0.5)
    xy = torch.stack([xs.float(), ys.float()], dim=1).to(device)

    K_test = tps_kernel(xy, pts1_world)
    P_test = torch.cat([torch.ones_like(xy[:, :1]), xy], dim=1)
    L_test = torch.cat([K_test, P_test], dim=1)

    xs2 = (L_test @ W_x).detach().cpu().numpy()
    ys2 = (L_test @ W_y).detach().cpu().numpy()

    dx = np.zeros((H, W), dtype=np.float32)
    dy = np.zeros((H, W), dtype=np.float32)
    dx[ys.cpu(), xs.cpu()] = xs2 - xs.cpu().numpy()
    dy[ys.cpu(), xs.cpu()] = ys2 - ys.cpu().numpy()

    return np.stack([dx, dy], axis=0)


def process_folder(input_folder, output_folder, device="cuda"):
    """对文件夹中每个 mask 视频进行多帧流计算"""
    os.makedirs(output_folder, exist_ok=True)

    for file in tqdm(sorted(os.listdir(input_folder))):
        if not file.endswith("-mask.mp4"):
            continue

        video_path = os.path.join(input_folder, file)
        frames = read_video_frames(video_path).to(device)
        T, H, W = frames.shape

        all_flows = []
        for i in range(T - 1):
            flow = compute_correspondence(frames[i], frames[i + 1], device=device)
            all_flows.append(flow)

        flows = np.stack(all_flows, axis=0)  # [T-1, 2, H, W]

        basename = os.path.splitext(file)[0].replace("-mask", "-map")
        out_path = os.path.join(output_folder, f"{basename}.npy")
        np.save(out_path, flows)

        print(f"✅ Saved: {out_path}, shape={flows.shape}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_folder = "./test-data/Gemini-out-expand-5"
    output_folder = input_folder+"-map"
    process_folder(input_folder, output_folder, device)
