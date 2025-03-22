import cv2
import random
import csv
import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import pipeline
from pytube import YouTube

# setting device for torch computations
device = 0 if torch.cuda.is_available() else -1

local_model_path = r"C:\Users\tanis\OneDrive\Desktop\tan priv\collegeTrade\local_model"
print("Using local model at:", local_model_path)

# Initialize the Hugging Face image-classification pipeline using the facial age model.
# This high-level pipeline takes care of image preprocessing and inference
age_pipe = pipeline("image-classification", model= local_model_path, device=device)

# Initialize MTCNN for face detection from different angles
mtcnn = MTCNN(keep_all=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def extract_frames(video_path):
    """

    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    # random offset between 0 and 10 to prevent foul play
    random_offset = random.uniform(0, 10)
    frame_times = []

    # for the first 5 mins, we capture frame every 10 sec
    t = random_offset
    while t < 300 and t < duration:
        frame_times.append(t)
        t += 10

    # after 5 mins, it is 30 secs till the end
    t = 300
    while t < duration:
        frame_times.append(t)
        t += 30

    extracted_frames = []
    for t in frame_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)  # setting video position
        ret, frame = cap.read()
        if ret:
            extracted_frames.append((t, frame))
    cap.release()
    return extracted_frames


def detect_faces_mtcnn(frame):
    """

    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(frame_rgb)

    if boxes is None:
        return []

    boxes_list = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        boxes_list.append((x1, y1, x2 - x1, y2 - y1))
    return boxes_list


def predict_age(face_img):
    """
    function to predict age using the hugging face pipeline
    """
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    # get predictions from the pipeline
    predictions = age_pipe(pil_img)

    # assuming for now that the first prediction is the chosen label
    predicted_label = predictions[0]['label']
    return predicted_label, predictions

def map_age_to_sentiment(age_str):
    """
    Define a helper function to map predicted age to a sentiment label.
    - "positive" if age < 16
    - "neutral" if age is between 16 and 18 (inclusive)
    - "negative" if age > 18
    """
    # Try converting the predicted label to an integer.
    try:
        age = int(age_str)
    except ValueError:
        # If the label is a range like "16-20", take the average.
        if "-" in age_str:
            parts = age_str.split("-")
            try:
                age = (int(parts[0]) + int(parts[1])) // 2
            except:
                age = 0
        else:
            age = 0
    if age < 16:
        return "positive"
    elif age >= 16 and age <= 20:
        return "neutral"
    else:
        return "negative"

# Helper function to download a YouTube video.
def download_youtube_video(url, output_path="videos", filename="downloaded_video.mp4"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    yt = YouTube(url)
    # Choose the highest resolution progressive stream (contains audio and video)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    video_path = stream.download(output_path=output_path, filename=filename)
    print(f"Downloaded video to: {video_path}")
    return video_path


def process_video(video_source, is_url=False, output_report="age_detection_report.csv", output_frames_dir="annotated_frames"):
    """
    # Main function that:
     1. Extracts frames from the video.
    2. Detects faces in each frame.
    3. Uses the pipeline to predict the age for each face.
    4. Maps the predicted age to a sentiment (positive/neutral/negative).
    5. Annotates the frame.
    6. Saves the annotated frames and writes a CSV report.
    """

    if is_url:
        video_path = download_youtube_video(video_source)
    else:
        video_path = video_source

    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    frames_with_time = extract_frames(video_path)
    print(f"Extracted {len(frames_with_time)} frames from the video.")

    results = []  # To store: [Timestamp, FaceIndex, x, y, w, h, PredictedAge, Sentiment]

    for frame_idx, (timestamp, frame) in enumerate(frames_with_time):
        boxes = detect_faces_mtcnn(frame)
        for face_idx, box in enumerate(boxes):
            x, y, w, h = box
            face_img = frame[y:y + h, x:x + w]

            # Predict age using the pipeline.
            predicted_label, _ = predict_age(face_img)
            sentiment = map_age_to_sentiment(predicted_label)

            # Annotate the frame with the bounding box and sentiment label.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{sentiment}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            results.append([timestamp, face_idx, x, y, w, h, predicted_label, sentiment])

        # Save the annotated frame.
        frame_filename = os.path.join(output_frames_dir, f"frame_{frame_idx:04d}_time_{int(timestamp)}s.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Processed frame at {timestamp:.2f}s with {len(boxes)} face(s) detected.")

    # Write the detection details to a CSV report.
    with open(output_report, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp(s)", "FaceIndex", "x", "y", "w", "h", "PredictedAge", "Sentiment"])
        writer.writerows(results)
    print(f"Report saved to {output_report}.")

if __name__ == "__main__":
    # Option 1: Process a local video file.
    # video_source = "your_local_video_file.mp4"
    # process_video(video_source, is_url=False)
    video_source = r"C:\Users\tanis\Videos\Captures\I am Lignite_ Kids of Lignite - Beulah Middle School - YouTube - Google Chrome 2025-03-20 17-49-36.mp4"
    process_video(video_source, is_url=False)

    # Option 2: Process a YouTube video by providing its URL.
    # video_url = "https://www.youtube.com/watch?v=7WB2qpTV404"  # Replace VIDEO_ID with an actual YouTube ID.
    # process_video(video_url, is_url=True)
