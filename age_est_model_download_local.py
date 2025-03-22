from huggingface_hub import snapshot_download

local_model_path = snapshot_download(
    repo_id="dima806/facial_age_image_detection",
    local_dir="local_model",
    local_dir_use_symlinks=False  # Avoid symlink issues on Windows
)
print("Model downloaded to:", local_model_path)