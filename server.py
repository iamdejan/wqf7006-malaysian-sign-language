import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import torch

import utils

labels = sorted(os.listdir(utils.TRAIN_DATASET_PATH)) if os.path.exists(utils.TRAIN_DATASET_PATH) else []


# Config
INPUT_SIZE = 258
HIDDEN_SIZE = 128
NUM_CLASSES = len(labels)
SEQ_LEN = 30
INPUT_SIZE = 258
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.35

# ---- Display options ----
DRAW_LANDMARKS = True  # True: draw MediaPipe pose+hands
SHOW_TOPK = 3  # 1: only top-1, >1: show top-k
PRED_EVERY_N_FRAMES = 2  # predict every N frames (speedup)
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2


# =========================
# Load label_map.json
# =========================
def load_gestures(label_map_path: str):
    label_map = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
    items = [(int(v), k) for k, v in label_map.items()]  # (idx, name)
    items.sort(key=lambda x: x[0])
    return [k for _, k in items]


LABEL_MAP_PATH = "label_map.json"
gestures = load_gestures(LABEL_MAP_PATH)
num_classes = len(gestures)
print("Loaded gestures:", num_classes)
print("First 10 gestures:", gestures[:10])


def get_device():
    return torch.device("cpu")


device = get_device()
print("Using device:", device)


model = utils.CustomLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
model.eval()
model.load_state_dict(torch.load("model/trained_model.pt", weights_only=True, map_location=device))  # Load your weights here


# =========================
# MediaPipe
# =========================
def mediapipe_detection(image_bgr, holistic):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)
    image_rgb.flags.writeable = True
    return results


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def extract_keypoints(results) -> np.ndarray:
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, lh, rh], axis=0)


# =========================
# Load norm stats (CRITICAL)
# =========================
norm_path = "norm_stats.npz"
if not os.path.exists(norm_path):
    raise FileNotFoundError(f"Missing {norm_path}. You MUST have it from training script.")
norm = np.load(norm_path)
feat_mean = norm["mean"].astype(np.float32)  # (1,1,258)
feat_std = norm["std"].astype(np.float32)  # (1,1,258)
print("✅ Loaded norm stats:", norm_path)


def normalize_seq_np(x_seq: np.ndarray) -> np.ndarray:
    return (x_seq - feat_mean) / feat_std


# =========================
# UI helpers
# =========================
def put_text_with_bg(img, text, org, scale=0.7, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    y0 = y - h - baseline - 4
    x0 = x
    x1 = x + w + 8
    y1 = y + 4
    y0 = max(0, y0)
    x1 = min(img.shape[1] - 1, x1)
    y1 = min(img.shape[0] - 1, y1)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


def format_topk(probs: torch.Tensor, k: int):
    k = min(k, probs.numel())
    vals, idxs = torch.topk(probs, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append((gestures[int(i)], float(v)))
    return out


def predict(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    sequence = deque(maxlen=SEQ_LEN)
    frame_id = 0

    # prediction cache
    last_pred = "..."
    last_prob = 0.0
    last_topk = []

    # FPS calc
    t_prev = time.time()
    fps_smooth = 0.0

    # output
    # Define output file name and codec (h264 for Gradio compatibility)
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"h264")

    # Get frame properties from your source (e.g., cap = cv2.VideoCapture(...))
    # Replace with actual dimensions and FPS if not using cv2.VideoCapture
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create the VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_id += 1

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            results = mediapipe_detection(frame, holistic)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        kp = extract_keypoints(results)
        sequence.append(kp)

        # predict (only when buffer full + every N frames)
        if len(sequence) == SEQ_LEN and (frame_id % PRED_EVERY_N_FRAMES == 0):
            x = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)  # (1,30,258)
            x = normalize_seq_np(x)
            xb = torch.tensor(x, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = model(xb)[0]
                probs = torch.softmax(logits, dim=0)

            last_topk = format_topk(probs, SHOW_TOPK)
            last_pred, last_prob = last_topk[0]

            # ✅ ONLY change: print final TOP-1 pred to console
            print(f"[frame={frame_id}] Pred: {last_pred} | prob={last_prob:.4f}")

        # FPS update (even when paused, keep stable)
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        inst_fps = 1.0 / dt
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps
        t_prev = t_now

        # overlay text
        x = 10
        y = 60
        y += 28
        put_text_with_bg(
            frame,
            f"Pred: {last_pred}  |  prob={last_prob:.3f}",
            (x, y),
            scale=TEXT_SCALE,
            thickness=TEXT_THICKNESS,
            text_color=(255, 255, 255),
            bg_color=(80, 20, 20),
            alpha=0.55,
        )
        y += 28

        if SHOW_TOPK > 1 and len(last_topk) > 1:
            for i, (name, p) in enumerate(last_topk[1:], start=2):
                put_text_with_bg(
                    frame,
                    f"Top-{i}: {name} ({p:.3f})",
                    (x, y),
                    scale=0.6,
                    thickness=2,
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    alpha=0.45,
                )
                y += 28

        # status line
        buf = len(sequence)
        if total_frames > 0:
            status = f"frame: {frame_id}/{total_frames} | buffer: {buf}/{SEQ_LEN} | FPS: {fps_smooth:.1f}"
        else:
            status = f"frame: {frame_id} | buffer: {buf}/{SEQ_LEN} | FPS: {fps_smooth:.1f}"
        put_text_with_bg(
            frame,
            status,
            (x, frame.shape[0] - 12),
            scale=0.6,
            thickness=2,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0),
            alpha=0.45,
        )

        out.write(frame)

    cap.release()
    out.release()
    return output_path  # Return the path to the saved video


with gr.Blocks() as demo:
    gr.Markdown("# Malaysia Sign Language")
    gr.Interface(predict, gr.Video(label=""), "playable_video", api_name="predict")


if __name__ == "__main__":
    demo.launch()
