import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
from scipy.interpolate import interp1d
from collections import defaultdict
import supervision as sv
import inference

# Load the car detection model
car_model = YOLO("yolo11x.pt")  # Replace with your YOLO model

# Load the left/right point detection model from Roboflow
point_model = inference.get_model("training-kkkrp/1", api_key="xxxxxxxxxxxx")  # Replace with your model details

# Known real-world distance in cm between left and right points
dist_bet_tracks = 100  # Adjust based on your calibration

# Replace the compute_scaling_factors function with fit_left_right_lines
def fit_left_right_lines(left_points, right_points):
    # Extract x and y coordinates
    left_y = np.array([pt[1] for pt in left_points])
    left_x = np.array([pt[0] for pt in left_points])
    right_y = np.array([pt[1] for pt in right_points])
    right_x = np.array([pt[0] for pt in right_points])

    # Fit interpolation functions
    left_line = interp1d(left_y, left_x, kind='cubic', bounds_error=False, fill_value="extrapolate")
    right_line = interp1d(right_y, right_x, kind='cubic', bounds_error=False, fill_value="extrapolate")

    return left_line, right_line

# Modify compute_real_distance to use left_line and right_line
def compute_real_distance(p1, p2, left_line, right_line, dist_bet_tracks):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    y_mid = (p1[1] + p2[1]) / 2

    # Interpolate x positions at y_mid
    x_left = left_line(y_mid)
    x_right = right_line(y_mid)

    # Compute pixel distance between lines at y_mid
    pixel_dist = x_right - x_left
    if pixel_dist == 0:
        scale = 0
    else:
        scale = dist_bet_tracks / pixel_dist  # cm per pixel

    # Pixel differences
    delta = p2 - p1

    # Real-world distance
    real_distance = np.linalg.norm(delta) * scale

    return real_distance, scale

# Function to detect cars and measure their dimensions
def detect_cars_in_frame_and_measure(frame, left_line, right_line, dist_bet_tracks):
    results = car_model(frame)
    annotated_frame = frame.copy()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id in [0, 2, 5] and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = {0: 'Person', 2: 'Car', 5: 'Bus'}[class_id]

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate real-world dimensions
                object_points = {
                    "x1": (x1, y1),
                    "x2": (x2, y1),
                    "x3": (x1, y2),
                    "x4": (x2, y2)
                }

                # Calculate width using x3 and x4, and get the scale
                width_real, scale = compute_real_distance(
                    object_points["x3"], object_points["x4"],
                    left_line, right_line, dist_bet_tracks
                )

                # Use the same scale for height calculation
                height_pixel = y2 - y1
                height_real = height_pixel * scale

                # Add dimensions to the label
                cv2.putText(annotated_frame, f'Width: {width_real:.2f} cm', (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Height: {height_real:.2f} cm', (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



                cv2.circle(annotated_frame, tuple(map(int, (x1,y1))), 5, (0, 0, 255), -1)  # Red circles for left points




    # Draw left and right interpolation lines
    frame_height = frame.shape[0]
    y_values = np.linspace(0, frame_height, num=500)
    left_x = left_line(y_values)
    right_x = right_line(y_values)

    # # Create points for polylines
    # left_points = np.vstack((left_x, y_values)).astype(int).T
    # right_points = np.vstack((right_x, y_values)).astype(int).T

    # # Draw the interpolation lines on the annotated frame
    # cv2.polylines(annotated_frame, [left_points], False, (255, 0, 0), 2)   # Blue line for left
    # cv2.polylines(annotated_frame, [right_points], False, (0, 0, 255), 2)  # Red line for right

    return annotated_frame

# Function to process and save the result video
def process_and_show_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # FFmpeg command for H.264 encoding
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{frame_width}x{frame_height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx264', output_video_path
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect left and right points using the Roboflow model
            result = point_model.infer(frame)[0]
            detections = sv.Detections.from_inference(result)

            left_points = []
            right_points = []

            # Process keypoints and categorize based on class_name
            for prediction in result.predictions:
                for keypoint in prediction.keypoints:
                    x = int(keypoint.x)
                    y = int(keypoint.y)
                    confidence = keypoint.confidence
                    class_name = keypoint.class_name

                    # Only process keypoints with confidence above threshold
                    if confidence > 0.5:
                        # Extract numeric part for sorting directly
                        point_info = (x, y, int(class_name.split('_')[1]))  # (x, y, class number)
                        
                        # Append to the correct list based on 'left' or 'right'
                        if 'left' in class_name:
                            left_points.append(point_info)
                        elif 'right' in class_name:
                            right_points.append(point_info)

            # Sort the points directly by the numeric part of class_name (class number)
            left_points.sort(key=lambda point: point[2])  # Sort by the third element, the numeric part
            right_points.sort(key=lambda point: point[2])  # Same for right points

            left_points = [(x, y) for x, y, _ in left_points]
            right_points = [(x, y) for x, y, _ in right_points]

            # Draw and annotate the points on the frame
            for left_point, right_point in zip(left_points, right_points):
                cv2.circle(frame, tuple(map(int, left_point)), 5, (0, 0, 255), -1)  # Red circles for left points
                cv2.circle(frame, tuple(map(int, right_point)), 5, (0, 255, 0), -1)  # Green circles for right points
                cv2.putText(frame, f'L {left_point}', (int(left_point[0]) + 10, int(left_point[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, f'R {right_point}', (int(right_point[0]) + 10, int(right_point[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                
            # Ensure we have enough points to compute scaling factors
            if len(left_points) >= 2 and len(right_points) >= 2:
                # Fit left and right lines
                left_line, right_line = fit_left_right_lines(left_points, right_points)

                # Detect objects and measure dimensions
                annotated_frame = detect_cars_in_frame_and_measure(frame, left_line, right_line, dist_bet_tracks)
            else:
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, 'Insufficient left/right points detected.',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Before writing the annotated frame
            cv2.putText(annotated_frame, f'dist_bet_tracks: {dist_bet_tracks} cm',
                        (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            process.stdin.write(annotated_frame.tobytes())

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()
        cap.release()
        print(f"Output video saved to: {output_video_path}")

# Run the function to process the video
input_video_path = 'qwerty2.mp4'  # Replace with your video path
output_video_path = 'final_output_boss_2.mp4'  # Output video path
process_and_show_video(input_video_path, output_video_path)
