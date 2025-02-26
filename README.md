# Application Documentation

## Installation

1. Clone the repository
```bash
git clone https://github.com/Scoutflo/ai-debugger.git
cd ai-debugger
```
2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
3. Install requirements
```bash
pip install -r requirements.txt
```

## Building with Docker

```bash
docker build -t my-python-app .
```

## Running the Application

```bash
docker run -p 8000:8000 my-python-app
```

## Environment Variables
The application can be configured using environment variables when running the Docker container. Example:
```bash
docker run -e VAR_NAME=value -p 8000:8000 my-python-app
```

## Dependencies
All required packages are listed in `requirements.txt` and automatically installed during the Docker build process.

## Project Structure Overview
```
.
├── Dockerfile
├── README.md
├── requirements.txt
├── app.py          # Main application entrypoint
└── (other project files)
```

The Dockerfile copies the entire directory structure into the container's `/app` working directory.

