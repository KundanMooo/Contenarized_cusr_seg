# syntax=docker/dockerfile:1

# 1) Base image with Python 3.9
FROM python:3.9-slim

# 2) Set a working directory in the container
WORKDIR /app

# 3) Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy all project files
COPY . .

# 5) Expose Streamlit default port
EXPOSE 8501

# 6) Default command to launch your Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
