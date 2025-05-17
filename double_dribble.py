
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import pygame

class DoubleDribbleDetector:
    def __init__(self):
        self.pose_model = YOLO("yolov8s-pose.pt")
        self.ball_model = YOLO("basketballModel.pt")
        self.cap = cv2.VideoCapture('double dribble.mp4')
        self.body_index = {"left_wrist": 9, "right_wrist": 10}
        self.hold_start_time = None
        self.is_holding = False
        self.was_holding = False
        self.hold_duration = 0.85
        self.hold_threshold_min = 50  # Minimum threshold for holding
        self.hold_threshold_max = 70  # Maximum threshold for holding
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        self.dribble_threshold = 18
        self.double_dribble_time = None
        self.double_dribble_triggered = False  # New flag to track double dribble trigger
        self.double_dribble_printed = False  # Flag to ensure double dribble is printed once

        # Add target size for resizing
        self.target_width = 850  # Reduced width for laptop screen
        self.scale_factor = None
        self.setup_frame_size()

        # Initialize pygame mixer for sound
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"Error initializing pygame mixer: {e}")

        # Load the buzzer sound
        self.buzzer_sound_path = "buzzer.mp3"  # Ensure this file is in the same directory as the script
        try:
            pygame.mixer.music.load(self.buzzer_sound_path)
        except Exception as e:
            print(f"Error loading sound file: {e}")

        # Define the codec and create VideoWriter object
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Create output directory if it doesn't exist
        output_dir = 'output_videos'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_video_path = os.path.join(output_dir, 'output_double_dribble.mp4')
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

    def setup_frame_size(self):
        # Get original frame size and calculate scaling
        original_width = int(self.cap.get(3))
        self.scale_factor = self.target_width / original_width
        self.frame_width = self.target_width

    def resize_frame(self, frame):
        # Resize frame while maintaining aspect ratio
        new_height = int(frame.shape[0] * self.scale_factor)
        return cv2.resize(frame, (self.target_width, new_height))

    def run(self):
        # Create the output directory for frames if it doesn't exist
        output_dir = 'double_dribble'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frame_counter = 0  # Counter to track frame numbers

        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                # Check if the frame is rotated and correct it
                height, width, _ = frame.shape
                if height < width:  # If the frame is wider than it is tall, rotate it
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # Resize frame before processing
                frame = self.resize_frame(frame)
                pose_annotated_frame, ball_detected = self.process_frame(frame)
                self.check_double_dribble()

                if (
                    self.double_dribble_time
                    and time.time() - self.double_dribble_time <= 3
                ):
                    red_tint = np.full_like(
                        pose_annotated_frame, (0, 0, 255), dtype=np.uint8
                    )
                    pose_annotated_frame = cv2.addWeighted(
                        pose_annotated_frame, 0.7, red_tint, 0.3, 0
                    )

                    cv2.putText(
                        pose_annotated_frame,
                        "Double dribble!",
                        (self.frame_width - 400, 100),  # Adjusted position for smaller frame
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,  # Reduced font size
                        (255, 255, 255),
                        3,  # Reduced thickness
                        cv2.LINE_AA,
                    )

                # Save the processed frame as an image
                frame_filename = os.path.join(output_dir, f"frame_{frame_counter:04d}.jpg")
                cv2.imwrite(frame_filename, pose_annotated_frame)
                frame_counter += 1

                # Display the frame
                cv2.namedWindow("Basketball Referee AI", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Basketball Referee AI", 700, 850)
                cv2.imshow("Basketball Referee AI", pose_annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        pose_annotated_frame = pose_results[0].plot()
        
        try:
            keypoints = pose_results[0].keypoints
            if keypoints.shape[0] > 0:
                left_wrist = keypoints.data[0][self.body_index["left_wrist"]][:2].cpu().numpy()
                right_wrist = keypoints.data[0][self.body_index["right_wrist"]][:2].cpu().numpy()
            else:
                print("No keypoints detected.")
                return pose_annotated_frame, False
        except Exception as e:
            print(f"Error processing keypoints: {e}")
            return pose_annotated_frame, False

        ball_results_list = self.ball_model(frame, verbose=False, conf=0.65)
        ball_detected = False

        for ball_results in ball_results_list:
            for bbox in ball_results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                ball_x_center = (x1 + x2) / 2
                ball_y_center = (y1 + y2) / 2

                self.update_dribble_count(ball_x_center, ball_y_center)
                self.prev_x_center = ball_x_center
                self.prev_y_center = ball_y_center
                ball_detected = True

                left_distance = np.hypot(
                    ball_x_center - left_wrist[0], ball_y_center - left_wrist[1]
                )
                right_distance = np.hypot(
                    ball_x_center - right_wrist[0], ball_y_center - right_wrist[1]
                )

                self.check_holding(left_distance, right_distance)

                cv2.rectangle(
                    pose_annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

                # Adjusted text size and position for smaller frame
                debug_info = [
                    f"Ball: ({ball_x_center:.2f}, {ball_y_center:.2f})",
                    f"Left Wrist: ({left_wrist[0]:.2f}, {left_wrist[1]:.2f})",
                    f"Right Wrist: ({right_wrist[0]:.2f}, {right_wrist[1]:.2f})",
                    f"Left Distance: {left_distance:.2f}",
                    f"Right Distance: {right_distance:.2f}",
                    f"Holding: {'Yes' if self.is_holding else 'No'}",
                    f"Dribble count: {self.dribble_count}"
                ]

                for i, text in enumerate(debug_info):
                    cv2.putText(
                        pose_annotated_frame,
                        text,
                        (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,  # Reduced font size
                        (0, 0, 0),
                        1,  # Reduced thickness
                        cv2.LINE_AA,
                    )

        if not ball_detected:
            self.hold_start_time = None
            self.is_holding = False

        return pose_annotated_frame, ball_detected

    def check_holding(self, left_distance, right_distance):
        # Update the hold condition to check if both distances are within the range of 90 to 120
        hold_threshold_min = 90
        hold_threshold_max = 120

        if (hold_threshold_min <= left_distance <= hold_threshold_max) and (hold_threshold_min <= right_distance <= hold_threshold_max):
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            elif time.time() - self.hold_start_time > self.hold_duration:
                self.is_holding = True
                self.was_holding = True
                self.dribble_count = 0
                self.double_dribble_triggered = False  # Reset the trigger flag
        else:
            if self.is_holding:
                self.was_holding = True  # Set was_holding to True when hold is released
            self.hold_start_time = None
            self.is_holding = False

    def update_dribble_count(self, x_center, y_center):
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center
            if (
                self.prev_delta_y is not None
                and delta_y < 0
                and self.prev_delta_y > self.dribble_threshold
            ):
                self.dribble_count += 1
            self.prev_delta_y = delta_y

    def check_double_dribble(self):
        if self.was_holding and self.dribble_count > 0 and not self.double_dribble_triggered:
            self.double_dribble_time = time.time()
            self.was_holding = False
            self.dribble_count = 0
            self.double_dribble_triggered = True  # Set the trigger flag
            print("Double dribble!")
            try:
                pygame.mixer.music.play()  # Play the buzzer sound
            except Exception as e:
                print(f"Error playing sound: {e}")

if __name__ == "__main__":
    detector = DoubleDribbleDetector()
    detector.run()