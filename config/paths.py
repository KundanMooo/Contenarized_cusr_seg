# config/paths.py
import os
from datetime import datetime

def get_paths():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "raw_upload": os.path.join("data", "uploaded", f"raw_{timestamp}.csv"),
        "processed_data": os.path.join("data", "processed", f"processed_{timestamp}.csv"),
        "model_path": os.path.join("models", f"model_{timestamp}.pkl")
    }
