# 🎬 Facial Age Estimation and Sentiment Analysis from Video to detect, prevent and report the usage of underage content on adult platforms

> **Detect faces, predict ages, and classify individuals by age group sentiment in videos (local and YouTube sources).**
> As of now, videos from Yt are down due to certain legal restrictions

---

## 🚀 Overview

This project analyzes video files—either from your local system or from YouTube—to detect human faces, estimate their ages using state-of-the-art deep learning models, and classify each person into sentiment categories based on age:

- ✅ **Positive**: **Under 16 years**
- ⚠️ **Neutral**: **Between 16–18 years**
- ❌ **Negative**: **Above 18 years**

The program extracts frames at specific intervals, detects faces robustly using the **MTCNN** algorithm, estimates ages using a pre-trained Vision Transformer (**ViT**) model from Hugging Face, and generates an easy-to-follow CSV report alongside annotated images.

---

## 🛠️ Key Features

- **Local & Online Videos**: Supports analysis from both local video files and YouTube links.
- **Robust Face Detection**: Reliable face detection from varied angles using **MTCNN**.
- **AI-based Age Prediction**: Powered by a state-of-the-art **Vision Transformer (ViT)**.
- **Sentiment Categorization**: Clearly defined categories based on age groups.
- **Visual Reporting**: Saves annotated frames for visual inspection.
- **Detailed CSV Reporting**: Generates a comprehensive CSV file containing timestamps, face coordinates, age predictions, and sentiments.

---

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/AI_visual_u-18_detection.git
cd AI_visual_u-18_detection
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
You can create requirements.txt by running:

bash
Copy
Edit
pip install torch torchvision opencv-python facenet-pytorch transformers pytube huggingface_hub numpy pillow
pip freeze > requirements.txt
```
🎯 Usage
Analyze Local Video
python
Copy
Edit
video_source = r"C:\path\to\your\video.mp4"
process_video(video_source, is_url=False)
Analyze YouTube Video
python
Copy
Edit
video_url = "https://www.youtube.com/watch?v=VIDEO_ID"
process_video(video_url, is_url=True)
📁 Project Structure
plaintext
Copy
Edit
facial-age-estimation/
├── annotated_frames/
│   └── (Annotated images saved here)
├── videos/
│   └── (Downloaded YouTube videos/local videos)
├── local_model/
│   └── (Pre-trained Hugging Face model files)
├── age_estimator.py
├── requirements.txt
├── README.md
└── LICENSE.md
🖼️ Example Outputs
After running, you will obtain:

Annotated Frames: Stored in annotated_frames/

CSV Report (age_detection_report.csv) with detailed predictions:

Timestamps

Face coordinates

Predicted age groups

Age-based sentiment (Positive, Neutral, Negative)

📚 Model Credits & References
Face Detection: MTCNN (facenet-pytorch)

Age Estimation: Facial Age Detection (ViT) - Hugging Face
