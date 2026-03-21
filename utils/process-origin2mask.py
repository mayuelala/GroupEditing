import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Optional
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
import random


# =====================
# 数据结构
# =====================
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(
            score=d["score"],
            label=d["label"],
            box=BoundingBox(
                xmin=d["box"]["xmin"],
                ymin=d["box"]["ymin"],
                xmax=d["box"]["xmax"],
                ymax=d["box"]["ymax"],
            ),
        )

# =====================
# 核心工具函数
# =====================
def refine_masks(masks: torch.BoolTensor):
    """SAM 输出掩码后处理"""
    masks = masks.cpu().float().mean(dim=1)  # [B,H,W]
    masks = (masks > 0).int().numpy().astype(np.uint8)
    return list(masks)

def detect(image: Image.Image, labels: List[str], threshold: float, detector):
    """使用 Grounding DINO 进行检测"""
    labels = [l if l.endswith(".") else l + "." for l in labels]
    results = detector(image, candidate_labels=labels, threshold=threshold)
    return [DetectionResult.from_dict(r) for r in results]

def segment(image: Image.Image, detections: List[DetectionResult], seg_model, processor):
    """使用 SAM 分割检测到的目标"""
    if not detections:
        return []
    boxes = [[d.box.xyxy for d in detections]]
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(seg_model.device)
    outputs = seg_model(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes,
    )[0]
    masks = refine_masks(masks)
    for det, mask in zip(detections, masks):
        det.mask = mask
    return detections

def grounded_segmentation(image: np.ndarray, labels: List[str],
                          threshold: float, detector, seg_model, processor):
    """对单帧整图进行检测和分割"""
    pil_img = Image.fromarray(image)
    detections = detect(pil_img, labels, threshold, detector)
    detections = segment(pil_img, detections, seg_model, processor)
    if not detections:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for det in detections:
        if det.mask is not None:
            combined = np.maximum(combined, det.mask * 255)
    return combined

# =====================
# 主执行逻辑
# =====================
if __name__ == "__main__":
    JSON_FILE_PATH = "./test-data/gemini-test.json"
    VIDEO_DIR = "./test-data/Gemini-out"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = "./models/IDEA-Research/grounding-dino-base"
    segmenter_id = "./models/facebook/sam-vit-huge"

    print(f"Loading models on {device}...")
    detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    seg_model = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    print("✅ Models loaded")

    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    random.shuffle(data)

    for item in data:
        video_file = item.get("image_filename").replace(".png","-origin.mp4")
        label = item.get("description", {}).get("item")
        if not video_file or not label:
            continue

        video_path = os.path.join(VIDEO_DIR, video_file)
        if not os.path.exists(video_path):
            print(f"⚠️ {video_path} not found")
            continue
        output_mask_filename = os.path.splitext(video_path)[0].replace("-origin","-mask.mp4")
        output_path = os.path.join(VIDEO_DIR, output_mask_filename)
        output_path = output_mask_filename
        print(f"Processing {video_path} -> {output_path}")

        if os.path.exists(output_path):
            print(f"⏭️ Skip existing mask: {output_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), False)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            mask = grounded_segmentation(frame, [label], 0.3, detector, seg_model, processor)
            out.write(mask)
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"  processed {frame_idx}/{total} frames")

        cap.release()
        out.release()
        print(f"✅ Saved mask video: {output_path}")

    print("🎉 All videos processed.")
