import os

import cv2
import gradio as gr
import mediapipe as mp
import torch

import utils

labels = sorted(os.listdir(utils.TRAIN_DATASET_PATH)) if os.path.exists(utils.TRAIN_DATASET_PATH) else []


# Config
INPUT_SIZE = 258
HIDDEN_SIZE = 64
NUM_CLASSES = len(labels)
SEQUENCE_LENGTH = 30

model = utils.CustomLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
model.load_state_dict(torch.load("model/trained_model.pt", weights_only=True))  # Load your weights here
model.eval()

# MediaPipe
# Initialize Holistic Model
holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def predict_stream(image, history_buffer):
    """
    image: The current video frame from Gradio
    history_buffer: A list holding the last 30 frames of keypoints
    """
    if image is None:
        return None, history_buffer

    # Initialize buffer if empty
    if history_buffer is None:
        history_buffer = []

    # 1. Preprocess Image (Color Conversion)
    # Gradio provides RGB, MediaPipe wants RGB.
    # We flip for mirror effect, but need to ensure it's writeable.
    image = cv2.flip(image, 1)

    # 2. Make Detections
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True

    # 3. Draw Landmarks (Visual Feedback)
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
    )
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
    )

    prediction_text = "Waiting for hands..."

    # 4. Filter Logic (Optional but recommended based on your training)
    # Only add to buffer if hands are detected, OR add zeros if you want continuous stream?
    # Your training code: `if results.left_hand_landmarks or results.right_hand_landmarks:`
    # We will replicate that logic:
    if results.left_hand_landmarks or results.right_hand_landmarks:
        # 5. Extract Keypoints (Your Custom Function)
        keypoints = utils.extract_keypoints(results)

        # 6. Add to History Buffer
        history_buffer.append(keypoints)

        # Maintain Sliding Window Size
        if len(history_buffer) > SEQUENCE_LENGTH:
            history_buffer = history_buffer[-SEQUENCE_LENGTH:]

        # 7. Predict
        if len(history_buffer) == SEQUENCE_LENGTH:
            # Prepare tensor: (1, 30, 258)
            input_seq = torch.tensor([history_buffer], dtype=torch.float64)

            with torch.no_grad():
                res = model(input_seq)
                idx = torch.argmax(res).item()

                # Safe Label Access
                if idx < NUM_CLASSES:
                    prediction_text = labels[idx]
                else:
                    prediction_text = f"Class {idx}"

    # Overlay Text
    h, w, _ = image.shape
    cv2.rectangle(image, (0, h - 40), (w // 2, h), (245, 117, 16), -1)
    cv2.putText(
        image,
        prediction_text,
        (15, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return image, history_buffer


with gr.Blocks() as demo:
    gr.Markdown("## Holistic Sign Language Recognition")
    gr.Markdown(f"Input Shape: {SEQUENCE_LENGTH} frames x {INPUT_SIZE} keypoints")

    with gr.Row():
        input_cam = gr.Image(sources=["webcam"], streaming=True, mirror_webcam=True)
        output_cam = gr.Image()

    state = gr.State([])

    # Stream event:
    # 'stream_every' controls how often we process frames (0.05s = 20 FPS).
    # Adjust this if the video lags.
    input_cam.stream(
        fn=predict_stream,
        inputs=[input_cam, state],
        outputs=[output_cam, state],
        stream_every=0.05,
    )


if __name__ == "__main__":
    demo.launch()
