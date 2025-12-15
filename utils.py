import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp

TRAIN_DATASET_PATH = r'./data/train_dataset'
MODEL_FOLDER_PATH = r"./model"
DROPOUT = 0.2

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion from BGR to RGB
    image.flags.writeable = False                   # Image is no longer writeable
    result = model.process(image)                   # Make prediction
    return result


def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)    # Draw right connections


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])

class ExtractLastTimeStep(nn.Module):
    def forward(self, x):
        # LSTM returns (output, (h_n, c_n))
        output, _ = x 
        # Extract the last time step: Shape (batch, seq, hidden) -> (batch, hidden)
        return output[:, -1, :]


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=DROPOUT, dtype=torch.float64).double(),

            # Custom layer to handle the output tuple and slice the last step
            ExtractLastTimeStep(),

            # Layer 1
            nn.Linear(hidden_size, 64, dtype=torch.float64),
            nn.BatchNorm1d(64, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            # Layer 2
            nn.Linear(64, 128, dtype=torch.float64),
            nn.BatchNorm1d(128, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            # Layer 3
            nn.Linear(128, 64, dtype=torch.float64),
            nn.BatchNorm1d(64, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            # Layer 4
            nn.Linear(64, 32, dtype=torch.float64),
            nn.BatchNorm1d(32, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(DROPOUT-0.1),

            # Layer 5
            nn.Linear(32, 32, dtype=torch.float64),
            # Batch norm here is optional, often skipped right before output
            nn.ReLU(),

            # Output Layer (No BN/Dropout/ReLU on the final logits)
            nn.Linear(32, num_classes, dtype=torch.float64),
        )

    def forward(self, x):
        return self.model(x)
