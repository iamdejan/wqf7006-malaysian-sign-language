import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import mediapipe as mp
import numpy as np

import utils

# silent mediapipe output
os.environ["GLOG_minloglevel"] = "2"  # '2' means suppress INFO and WARNING
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GOOGLE_LOG_MIN_SEVERITY"] = "2"  # This can also be used

# --- Setup Paths ---
train_dataset_path = r"./data/train_dataset"
video_directory = r"./data/video_dataset"
trained_gestures = sorted(os.listdir(train_dataset_path)) if os.path.exists(train_dataset_path) else []

gestures_files = sorted(os.listdir(video_directory))
remaining_list = sorted([item for item in gestures_files if item not in trained_gestures])

print(f"Gestures to process: {len(remaining_list)}")


# --- Worker Function for Parallel Processing ---
def process_single_video(task_data):
    """
    Processes a single video file.
    Args:
        task_data: tuple containing (gesture_name, video_filename, source_video_path, target_landmark_folder)
    """
    gesture_name, video_filename, src_path, target_folder = task_data

    cv2.setNumThreads(0)

    # Create the target folder safely (exist_ok prevents errors in parallel)
    os.makedirs(target_folder, exist_ok=True)

    cap = cv2.VideoCapture(src_path)
    count = 0
    frame_count = 0

    # Initialize MediaPipe per video (needed for parallel safety)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            # Stop if video ends or we reach 30 frames
            if not ret or frame_count == 30:
                break

            # 1. Detection (KEEP THIS)
            # Assuming mediapipe_detection is defined in your global scope or imported
            results = utils.mediapipe_detection(frame, holistic)

            # 2. Save Data
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # assuming extract_keypoints is defined
                keypoints = utils.extract_keypoints(results)

                npy_path = os.path.join(target_folder, str(count + 1))
                np.save(npy_path, keypoints)

                frame_count += 1

            count += 1

    cap.release()
    return f"Done: {gesture_name}/{video_filename}"


def main():
    # 1. Prepare a flat list of all tasks to balance load
    #    (Processing by list of videos is better than list of gestures)
    all_tasks = []

    print("Gathering video files...")
    for ges in remaining_list:
        data_path = os.path.join(video_directory, ges)
        if not os.path.exists(data_path):
            continue

        data_video = os.listdir(data_path)

        for vid in data_video:
            # Prepare paths ahead of time
            source_path = os.path.join(video_directory, ges, vid)
            # Note: Adding 'landmarks' prefix to folder name as per your original code logic
            target_path = os.path.join(train_dataset_path, ges, "landmarks" + vid)

            # Pack data into a tuple
            all_tasks.append((ges, vid, source_path, target_path))

    print(f"Total videos to process: {len(all_tasks)}")

    # 2. Run Parallel Processing
    #    Adjust max_workers if your PC freezes (e.g., set to os.cpu_count() - 2)
    max_workers = os.cpu_count()

    print(f"Starting extraction with {max_workers} CPU cores...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        results = executor.map(process_single_video, all_tasks)

        # Optional: Print progress
        i = 0
        for item in results:
            if i % 10 == 0:
                print(f"Processed {i}/{len(all_tasks)} videos...", end="\r")
            print(item)
            i += 1

    print("\n\nAll extraction completed!")


# --- Main Execution ---
if __name__ == "__main__":
    main()
