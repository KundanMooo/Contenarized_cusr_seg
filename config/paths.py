import os
from datetime import datetime

def get_paths():
    """Return absolute paths for all file locations"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return {
        "raw_upload": os.path.join(base_dir, "data", "uploaded", f"raw_{timestamp}.csv"),
        "processed_data": os.path.join(base_dir, "data", "processed", f"processed_{timestamp}.csv"),
        "model_path": os.path.join(base_dir, "models", f"model_{timestamp}.pkl"),
        "base_dir": base_dir
    }