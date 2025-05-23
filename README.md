#  AI-assistance-to-basketball-referees

This project is an AI-powered assistant designed to support basketball referees by detecting player actions and rule violations in real time. The system uses advanced computer vision techniques such as YOLOv8 for detecting players, the ball, and court boundaries, along with pose estimation to track body keypoints and understand player movements.

By analyzing pose keypoints, the system can recognize actions like dribbling, holding, and traveling. It then applies basketball rules to automatically identify violations such as double dribble, traveling, and 24-second shot clock violations. Team identification is handled through jersey color and number recognition, helping to monitor player actions more precisely.
Since the ball detection model is large in size, it has not been included in this repository. You can download it from the following https://drive.google.com/file/d/1_CIPPY2FFcikNVwCFXK8e_QujTjNI6Ya/view?usp=drive_link
