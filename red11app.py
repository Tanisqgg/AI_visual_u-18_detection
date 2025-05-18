from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import csv
from age_estimator import process_video  # Import your existing processing function

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a strong secret key

# Configure upload and output folders
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
PROCESSED_FOLDER = os.path.join(os.getcwd(), "annotated_frames")
REPORT_FILE = os.path.join(os.getcwd(), "age_detection_report.csv")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    # Allow only common video formats
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov', 'avi'}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["video"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(video_path)
            try:
                process_video(video_path, is_url=False, output_report=REPORT_FILE, output_frames_dir=PROCESSED_FOLDER)
                flash("Video processed successfully!")
            except Exception as e:
                flash(f"An error occurred while processing the video: {e}")
                return redirect(request.url)
            return redirect(url_for("results"))
    # Render the custom index template
    return render_template("red11index.html")

@app.route("/results")
def results():
    # Gather annotated frames and check if the report exists.
    report_exists = os.path.exists(REPORT_FILE)
    frames = []
    if os.path.exists(PROCESSED_FOLDER):
        frames = sorted(os.listdir(PROCESSED_FOLDER))

    # Read the CSV report to generate summary stats.
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    if report_exists:
        with open(REPORT_FILE, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sentiment = row["Sentiment"]
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1

    return render_template("red11results.html",
                           report_exists=report_exists,
                           report_file="age_detection_report.csv",
                           frames=frames,
                           sentiment_counts=sentiment_counts)

@app.route("/download_report")
def download_report():
    return send_from_directory(os.getcwd(), "age_detection_report.csv", as_attachment=True)

@app.route("/frames/<filename>")
def serve_frame(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
