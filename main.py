import gradio as gr
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

LABELS = ["ribut", "nasi_lemak", "pandai", "panas", "baik", "bila", "tandas", "apa", "beli_2", "hari", "anak_lelaki", "panas_2", "beli", "hi", "marah", "boleh", "assalamualaikum", "apa_khabar", "tidur", "masalah", "abang", "polis", "perlahan_2", "perlahan", "saudara", "siapa", "bagaimana", "bahasa_isyarat", "baik_2", "bapa_saudara", "berapa", "hujan", "kakak", "keluarga", "mana", "payung", "perempuan", "lelaki", "curi", "berlari", "sampai", "mari", "pergi_2", "emak", "ada", "mohon", "kereta", "suka", "ayah", "main", "buang", "lemak", "minum", "bomba", "pukul", "buat", "bawa", "tanya", "anak_perempuan", "sejuk", "kacau", "ambil", "pensil", "emak_saudara", "teh_tarik", "berjalan", "sudah", "lupa", "jahat", "tolong", "bola", "bas", "masa", "baca", "kesakitan", "pandai_2", "jumpa", "dapat", "arah", "teksi", "dari", "jam", "sekolah", "jangan", "nasi", "makan", "bapa", "pergi", "pinjam", "pen"]

# ==========================================
# 1. SETUP MEDIAPIPE HOLISTIC (Matches your Preprocessing)
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Use the exact extraction logic from your training script
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatenate exactly as you did in training
    return np.concatenate([pose, lh, rh])


# ==========================================
# 2. DEFINE MODEL (Updated Input Size)
# ==========================================
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output_layer(x)
        return x

# Config
INPUT_SIZE = 258
HIDDEN_SIZE = 64
NUM_CLASSES = len(LABELS)
SEQUENCE_LENGTH = 30

model = CustomLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
model.load_state_dict(torch.load("model/trained_model.pt", weights_only=True)) # Load your weights here
model.eval()

# MediaPipe
# Initialize Holistic Model
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# ==========================================
# 3. REAL-TIME PROCESSING FUNCTION
# ==========================================
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
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    prediction_text = "Waiting for hands..."

    # 4. Filter Logic (Optional but recommended based on your training)
    # Only add to buffer if hands are detected, OR add zeros if you want continuous stream?
    # Your training code: `if results.left_hand_landmarks or results.right_hand_landmarks:`
    # We will replicate that logic:
    if results.left_hand_landmarks or results.right_hand_landmarks:
        
        # 5. Extract Keypoints (Your Custom Function)
        keypoints = extract_keypoints(results)
        
        # 6. Add to History Buffer
        history_buffer.append(keypoints)
        
        # Maintain Sliding Window Size
        if len(history_buffer) > SEQUENCE_LENGTH:
            history_buffer = history_buffer[-SEQUENCE_LENGTH:]

        # 7. Predict
        if len(history_buffer) == SEQUENCE_LENGTH:
            # Prepare tensor: (1, 30, 258)
            input_seq = torch.tensor([history_buffer], dtype=torch.float32)
            
            with torch.no_grad():
                res = model(input_seq)
                idx = torch.argmax(res).item()
                
                # Safe Label Access
                if idx < NUM_CLASSES:
                    prediction_text = LABELS[idx]
                else:
                    prediction_text = f"Class {idx}"

    # Overlay Text
    h, w, _ = image.shape
    cv2.rectangle(image, (0, h-40), (w // 2, h), (245, 117, 16), -1)
    cv2.putText(image, prediction_text, (15, h-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image, history_buffer


# ==========================================
# 4. GRADIO INTERFACE
# ==========================================
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
        stream_every=0.1
    )


if __name__ == "__main__":
    demo.launch()
