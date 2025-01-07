
# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create a Python virtual environment
RUN python -m venv /app/venv

# Activate the virtual environment and install required packages
RUN . /app/venv/bin/activate && pip install --no-cache-dir -r requirements.txt


# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV PATH="/app/venv/bin:$PATH"
ENV FLASK_APP=main.py

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
