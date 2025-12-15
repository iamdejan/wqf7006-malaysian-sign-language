import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

train_dataset_path = r'./data/train_dataset'
gestures = sorted(os.listdir(train_dataset_path)) if os.path.exists(train_dataset_path) else []
model_folder_path = r"./model"


# Helper function to process ONE gesture (to be run in parallel)
def process_single_gesture(gs, train_dataset_path, label_map):
    gesture_videos = []
    gesture_labels = []
    
    # Path to the specific gesture folder
    gs_path = os.path.join(train_dataset_path, gs)
    if not os.path.exists(gs_path):
        return [], []

    # Get list of video folders (fname)
    video_folders = [f for f in os.listdir(gs_path) if os.path.isdir(os.path.join(gs_path, f))]

    for video_folder in video_folders:
        load_path = os.path.join(gs_path, video_folder)
        
        # Get all .npy files and sort them numerically
        npy_files = [f for f in os.listdir(load_path) if f.endswith('.npy')]
        # Optimization: Filter digits once to avoid repeated calls
        npy_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if not npy_files:
            continue

        # Load all frames for this video
        # Optimization: List comprehension is slightly faster than append loop
        video_frames = [np.load(os.path.join(load_path, npy)) for npy in npy_files]
        
        gesture_videos.append(video_frames)
        gesture_labels.append(label_map[gs])

    return gesture_videos, gesture_labels

def load_and_pad_data(gestures, train_dataset_path, label_map, model_folder_path):
    all_sequences = []
    all_labels = []

    # 1. PARALLEL LOADING
    print(f"Starting parallel load for {len(gestures)} gestures...")

    # Determine number of workers (usually number of CPU cores)
    max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function to pass constant arguments
        worker = partial(process_single_gesture, 
                         train_dataset_path=train_dataset_path, 
                         label_map=label_map)

        # Map the worker to the gestures
        results = executor.map(worker, gestures)

        # Collect results as they finish
        for videos, labels in results:
            all_sequences.extend(videos)
            all_labels.extend(labels)

    print(f"Loading complete. Total sequences: {len(all_sequences)}")

    if not all_sequences:
        print("No data found.")
        return

    # 2. VECTORIZED PADDING (Much faster than list comprehension)
    # Find global dimensions
    max_seq_len = max(len(seq) for seq in all_sequences)
    feature_dim = all_sequences[0][0].shape[0] # Assuming all frames have same feature dim
    num_samples = len(all_sequences)

    print(f"Padding data. Shape: ({num_samples}, {max_seq_len}, {feature_dim})")

    # Pre-allocate one giant array of zeros (Masking/Padding built-in)
    X = np.zeros((num_samples, max_seq_len, feature_dim), dtype=np.float32)
    
    # Fill the array
    for i, seq in enumerate(all_sequences):
        length = len(seq)
        X[i, :length, :] = np.array(seq)

    y = np.array(all_labels)

    # 3. SAVE ONCE (Or modify to save per batch if memory is tight)
    print("Saving to disk...")
    np.save(f'{model_folder_path}/X_all.npy', X)
    np.save(f'{model_folder_path}/y_all.npy', y)
    print("Done!")


def main():
    label_map = {label: num for num, label in enumerate(gestures)}
    load_and_pad_data(gestures, train_dataset_path, label_map, model_folder_path)


# --- usage ---
if __name__ == '__main__':
    main()
