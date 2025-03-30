# Use a base image
FROM python:3.11  

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    default-libmysqlclient-dev \
    build-essential  

# Copy and install dependencies
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  

# Copy the rest of the application
COPY . .  

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit without file watching (fixes some Docker issues)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=false"]
