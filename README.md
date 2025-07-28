#  AI ASSISTANCE SYSTEM FOR BASKETBALL REFEREES

This project introduces an AI-powered assistant developed to support basketball referees by automatically detecting player actions and rule violations in real time. The system leverages advanced computer vision techniques to ensure accurate and efficient decision-making during gameplay.

## Overview
The assistant integrates object detection using YOLOv8 to identify players, the ball, and court boundaries. It also incorporates pose estimation to track body keypoints, enabling detailed analysis of player movements and behaviors.

By combining object detection and pose-based action recognition, the system can interpret in-game events and apply basketball rules to identify potential violations.

## Key Features
### Real-Time Object Detection
Player Detection: Identifies and tracks players throughout the game.

Ball Detection: Locates and follows the basketball in real time.

Court Boundary Detection: Recognizes court landmarks such as the half-court line, three-point arc, and paint area.

### Action Recognition Using Pose Estimation
Uses keypoint tracking to analyze body movements.

Detects and classifies player actions including:

Dribbling

Holding

Traveling

### Automatic Rule Violation Detection
Applies predefined basketball rule logic to detect:

Double dribble

Traveling violations

24-second shot clock violations

### Team and Player Identification
Recognizes team affiliation using jersey color segmentation.

Identifies individual players by analyzing jersey numbers for accurate action attribution.

## Ball Detection Model
Due to file size constraints, the ball detection model is not included in this repository.
You can download the pre-trained ball detection model from the following link:
Download Ball Detection Model
 https://drive.google.com/file/d/1_CIPPY2FFcikNVwCFXK8e_QujTjNI6Ya/view?usp=drive_link


## Future Work
Integration with digital shot clocks and scoreboard systems

Foul detection through body contact and proximity analysis

Optimized deployment on edge devices (e.g., Jetson Nano, Raspberry Pi)

Development of a user interface for referees and coaches

## Contributing
Contributions are welcome. You may:

Open issues to report bugs or suggest features

Submit pull requests with improvements or enhancements

Engage in discussions to shape future development
