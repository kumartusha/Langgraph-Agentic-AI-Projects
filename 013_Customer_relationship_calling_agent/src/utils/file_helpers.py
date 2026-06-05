import os
import shutil
from fastapi import UploadFile

UPLOAD_DIR = "temp_uploads"

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """
    Saves an uploaded file to a temporary directory and returns the absolute path.
    """
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        
    file_path = os.path.join(UPLOAD_DIR, upload_file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    return os.path.abspath(file_path)

def cleanup_tmp_file(file_path: str):
    """
    Deletes the temporary file after processing.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Failed to clean up temp file {file_path}: {str(e)}")
