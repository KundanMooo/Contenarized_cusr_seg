import os

def get_paths():
    """
    Returns a dict of all I/O paths.
    """
    base = os.getcwd()
    return {
        # where Streamlit writes your uploaded CSV
        "raw_upload": os.path.join(base, "data", "raw_upload.csv"),
        # where processed CSV will be saved
        "processed_data": os.path.join(base, "data", "processed_data.csv"),
        # where the trained model will be pickled
        "model": os.path.join(base, "models", "model.pkl"),
    }
