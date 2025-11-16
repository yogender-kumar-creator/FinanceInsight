# -----------------------------------------------------
# Professional Streamlit App Deployment (Docker)
# -----------------------------------------------------

FROM python:3.10

# Set working directory inside container
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application
COPY . .

# Set Streamlit to run on container's 0.0.0.0
EXPOSE 8501

# Start Streamlit App
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
