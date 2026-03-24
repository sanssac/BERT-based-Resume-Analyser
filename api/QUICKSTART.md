# Quick Start Guide - Resume Analyser API

## Start the API in 3 Steps

### Step 1: Activate Environment
```powershell
# Windows PowerShell
F:/anaconda/envs/DL-GPU/python.exe
```

### Step 2: Start the Server
```powershell
cd api
F:/anaconda/envs/DL-GPU/python.exe app.py
```

### Step 3: Access the API
Open your browser and visit:
- **Interactive Docs:** http://localhost:8000/docs
- **API Root:** http://localhost:8000

## Test the API

### Using the Browser (Swagger UI)
1. Go to http://localhost:8000/docs
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Paste a resume text
5. Click "Execute"

### Using Python
```python
import requests

resume = "Python developer with 5 years experience in Django and Flask"
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": resume}
)
print(response.json())
```

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Python developer with 5 years experience"}'
```

## Run Tests
```powershell
cd api
F:/anaconda/envs/DL-GPU/python.exe test_api.py
```

## Stop the API
Press `Ctrl+C` in the terminal where the API is running.
