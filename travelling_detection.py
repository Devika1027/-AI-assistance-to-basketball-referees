
import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import os
import time
import pygame

# Load the YOLO models
ball_model = YOLO("basketballModel.pt")
pose_model = YOLO("yolov8s-pose.pt")

# Open the video file
cap = cv2.VideoCapture('group travelling.mp4')

# Initialize counters and positions
dribble_count = 0
step_count = 0
prev_x_center = None
prev_y_center = None
prev_left_ankle_y = None
prev_right_ankle_y = None
prev_delta_y = None
ball_not_detected_frames = 0
max_ball_not_detected_frames = 20  # Adjust based on your requirement
dribble_threshold = 18  # Adjust based on observations
step_threshold = 5
min_wait_frames = 7
wait_frames = 0
travel_detected = False
travel_timestamp = None
total_dribble_count = 0
total_step_count = 0
ball_handler_id = None

# Define the body part indices
body_index = {"left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16, "left_wrist": 9, "right_wrist": 10}

# Define the frame dimensions and fps
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_video_path = 'output_travelling_detection.avi'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Define target width for resizing
target_width = 850
scale_factor = target_width / frame_width

def resize_frame(frame):# Resize frame while maintaining aspect ratio
    new_height = int(frame.shape[0] * scale_factor)
    return cv2.resize(frame, (target_width, new_height))

# Make directory to save travel footage
if not os.path.exists("travel_footage"):
    os.makedirs("travel_footage")

# Initialize frame buffer and frame saving settings
frame_buffer = deque(maxlen=30)  # Buffer to hold frames
save_frames = 60  # Number of frames to save after travel is detected
frame_save_counter = 0
saving = False

# Initialize pygame mixer
pygame.mixer.init()

# Load the buzzer sound
buzzer_sound_path = "buzzer.mp3"  # Ensure this file is in the same directory as the script
pygame.mixer.music.load(buzzer_sound_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Resize frame before processing
        frame = resize_frame(frame)

        # Append the frame to buffer
        frame_buffer.append(frame)

        # Ball detection
        ball_results_list = ball_model(frame, verbose=False, conf=0.65)

        ball_detected = False

        for results in ball_results_list:
            for bbox in results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]

                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                if prev_y_center is not None:
                    delta_y = y_center - prev_y_center

                    if (
                        prev_delta_y is not None
                        and prev_delta_y > dribble_threshold
                        and delta_y < -dribble_threshold
                    ):
                        dribble_count += 1
                        total_dribble_count += 1

                    prev_delta_y = delta_y

                prev_x_center = x_center
                prev_y_center = y_center

                ball_detected = True
                ball_not_detected_frames = 0

            annotated_frame = results.plot()

        # Increment the ball not detected counter if ball is not detected
        if not ball_detected:
            ball_not_detected_frames += 1

        # Reset step count if ball is not detected for a prolonged period
        if ball_not_detected_frames >= max_ball_not_detected_frames:
            step_count = 0

        # Pose detection
        pose_results = pose_model(frame, verbose=False, conf=0.5)

        if pose_results and pose_results[0].keypoints is not None:
            # Convert keypoints to a NumPy array safely
            keypoints_array = pose_results[0].keypoints.data.cpu().numpy()

            if keypoints_array.size > 0:
                # Round the results to the nearest decimal
                rounded_results = np.round(keypoints_array, 1)

                try:
                    for person_id, keypoints in enumerate(rounded_results):
                        left_knee = keypoints[body_index["left_knee"]]
                        right_knee = keypoints[body_index["right_knee"]]
                        left_ankle = keypoints[body_index["left_ankle"]]
                        right_ankle = keypoints[body_index["right_ankle"]]
                        left_wrist = keypoints[body_index["left_wrist"]]
                        right_wrist = keypoints[body_index["right_wrist"]]

                        if (
                            (left_knee[2] > 0.5)
                            and (right_knee[2] > 0.5)
                            and (left_ankle[2] > 0.5)
                            and (right_ankle[2] > 0.5)
                        ):
                            if ball_handler_id is None:
                                # Check if this person is the ball handler
                                left_distance = np.hypot(
                                    x_center - left_wrist[0], y_center - left_wrist[1]
                                )
                                right_distance = np.hypot(
                                    x_center - right_wrist[0], y_center - right_wrist[1]
                                )
                                if left_distance < 50 or right_distance < 50:
                                    ball_handler_id = person_id

                            if person_id == ball_handler_id:
                                if (
                                    prev_left_ankle_y is not None
                                    and prev_right_ankle_y is not None
                                    and wait_frames == 0
                                    and not travel_detected  # Skip step counting if travel detected
                                ):
                                    left_diff = abs(left_ankle[1] - prev_left_ankle_y)
                                    right_diff = abs(right_ankle[1] - prev_right_ankle_y)

                                    # Count steps only when the ball handler's feet touch the ground
                                    if max(left_diff, right_diff) > step_threshold and left_ankle[2] > 0.5 and right_ankle[2] > 0.5:
                                        step_count += 1
                                        total_step_count += 1
                                        if total_step_count <= 3:
                                            print(f"Step taken: {total_step_count}")
                                            cv2.putText(
                                                frame,
                                                f"Step taken: {total_step_count}",
                                                (50, 100 + total_step_count * 50),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (0, 0, 0),
                                                2,
                                                cv2.LINE_AA,
                                            )
                                        wait_frames = min_wait_frames  # Update wait_frames

                                prev_left_ankle_y = left_ankle[1]
                                prev_right_ankle_y = right_ankle[1]

                                if wait_frames > 0:
                                    wait_frames -= 1

                except:
                    print("No human detected.")

        pose_annotated_frame = pose_results[0].plot()

        # Combining frames
        combined_frame = cv2.addWeighted(
            frame, 0.6, pose_annotated_frame, 0.4, 0
        )

        # Drawing counts on the frame
        cv2.putText(
            combined_frame,
            f"Dribble count: {total_dribble_count}",
            (50, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        # Travel detection
        if ball_detected and total_step_count >= 3:
            print("Travel detected!")
            pygame.mixer.music.play()  # Play the buzzer sound
            step_count = 0  # Reset step count
            total_step_count = 0  # Reset total step count
            travel_detected = True
            travel_timestamp = time.time()

        if travel_detected:
            # Change the tint of the frame and write text if travel was detected
            red_tint = np.full_like(combined_frame, (0, 0, 255), dtype=np.uint8)  # Red tint
            combined_frame = cv2.addWeighted(combined_frame, 0.7, red_tint, 0.3, 0)
            cv2.putText(
                combined_frame,
                "Travel Detected!",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )

        # Reset counts when a dribble is detected
        if dribble_count > 0:
            step_count = 0
            dribble_count = 0

        # Write the frame to the output video
        out.write(combined_frame)

        cv2.imshow("Travel Detection", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
