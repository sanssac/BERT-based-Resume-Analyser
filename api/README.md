# Resume Analyser API

FastAPI-based REST API for classifying resumes into 25 job categories using the fine-tuned BERT model.

## Model Performance

- **Accuracy**: 95.85%
- **Balanced Accuracy**: 95.66%
- **F1-Score (Macro)**: 95.65%
- **Configuration**: config_6_moderate (3 epochs, optimized regularization)
- **Training Time**: 4.37 minutes

## Features

- 🚀 Fast and efficient resume classification
- 📊 Single and batch prediction endpoints
- 🎯 Returns category with confidence scores
- 📝 Automatic API documentation (Swagger/ReDoc)
- 🔄 CORS enabled for web applications
- 💻 Supports both CPU and GPU inference

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

1. **Navigate to API directory**
```bash
cd api
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

3. **Run the API**
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Check if the model is loaded and API is healthy.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "categories_count": 25,
  "timestamp": "2025-11-03T12:00:00"
}
```

### 3. Get Categories
```
GET /categories
```
Get all 25 available job categories.

**Response:**
```json
{
  "total": 25,
  "categories": {
    "0": "Data Science",
    "1": "Java Developer",
    ...
  }
}
```

### 4. Single Resume Prediction
```
POST /predict
```

**Request Body:**
```json
{
  "text": "Skills: Python, Machine Learning, Deep Learning, PyTorch, TensorFlow. Experience: 3 years in Data Science and AI model development."
}
```

**Query Parameters:**
- `include_all_scores` (boolean, optional): Include confidence scores for all categories

**Response:**
```json
{
  "category": "Data Science",
  "category_id": 0,
  "confidence": 0.98,
  "all_scores": null
}
```

### 5. Batch Resume Prediction
```
POST /batch-predict
```

**Request Body:**
```json
{
  "resumes": [
    "Python developer with 5 years experience in Django and Flask...",
    "Experienced data scientist with expertise in machine learning...",
    "Java backend developer skilled in Spring Boot..."
  ]
}
```

**Query Parameters:**
- `include_all_scores` (boolean, optional): Include confidence scores for all categories

**Response:**
```json
{
  "predictions": [
    {
      "category": "Python Developer",
      "category_id": 4,
      "confidence": 0.95,
      "all_scores": null
    },
    {
      "category": "Data Science",
      "category_id": 0,
      "confidence": 0.97,
      "all_scores": null
    },
    {
      "category": "Java Developer",
      "category_id": 1,
      "confidence": 0.93,
      "all_scores": null
    }
  ],
  "total_processed": 3
}
```

## Usage Examples

### Python (requests)

```python
import requests

# API URL
API_URL = "http://localhost:8000"

# Single prediction
resume_text = """
Skills: Python, Machine Learning, Deep Learning, PyTorch, TensorFlow
Experience: 3 years in Data Science and AI model development
Projects: Built recommendation systems, sentiment analysis models
Education: Masters in Computer Science
"""

response = requests.post(
    f"{API_URL}/predict",
    json={"text": resume_text}
)

result = response.json()
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### JavaScript (fetch)

```javascript
const API_URL = "http://localhost:8000";

// Single prediction
const resumeText = `
Skills: Python, Machine Learning, Deep Learning, PyTorch, TensorFlow
Experience: 3 years in Data Science and AI model development
`;

fetch(`${API_URL}/predict`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ text: resumeText })
})
.then(response => response.json())
.then(data => {
  console.log(`Category: ${data.category}`);
  console.log(`Confidence: ${(data.confidence * 100).toFixed(2)}%`);
})
.catch(error => console.error('Error:', error));
```

### cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Python developer with 5 years experience in Django and Flask"
  }'

# Batch prediction
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "resumes": [
      "Python developer with 5 years experience",
      "Data scientist with ML expertise"
    ]
  }'

# Get all scores
curl -X POST "http://localhost:8000/predict?include_all_scores=true" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Python developer with 5 years experience"
  }'
```

## Interactive Documentation

Once the API is running, visit:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide interactive documentation where you can test the API directly from your browser.

## Job Categories

The API classifies resumes into these 25 categories:

1. Data Science
2. Java Developer
3. Testing
4. DevOps Engineer
5. Python Developer
6. Web Developer
7. HR
8. Hadoop
9. Blockchain
10. ETL Developer
11. Operations Manager
12. Sales
13. Mechanical Engineer
14. Arts
15. Database
16. Electrical Engineering
17. Health and Fitness
18. PMO
19. Business Analyst
20. DotNet Developer
21. Automation Testing
22. Network Security Engineer
23. SAP Developer
24. Civil Engineer
25. Advocate

## Performance

- **Model:** BERT-base-uncased (109M parameters)
- **Inference Time:** ~50-100ms per resume (GPU) / ~200-500ms (CPU)
- **Batch Processing:** Recommended for processing multiple resumes
- **Max Input Length:** 200 tokens

## Configuration

### Environment Variables

You can configure the API using environment variables:

```bash
# Port (default: 8000)
export PORT=8000

# Host (default: 0.0.0.0)
export HOST=0.0.0.0

# Model path (default: loads from Hugging Face)
export MODEL_PATH=SwaKyxd/resume-analyser-bert

# Device (default: auto-detect)
export DEVICE=cuda  # or cpu
```

### Production Deployment

For production, consider:

1. **Use Gunicorn with Uvicorn workers:**
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. **Configure CORS properly:**
Edit the `allow_origins` in `app.py` to specify allowed domains

3. **Add authentication:**
Implement API key or JWT authentication for security

4. **Use a reverse proxy:**
Deploy behind Nginx or Apache for better performance

5. **Enable HTTPS:**
Use SSL certificates for secure communication

## Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t resume-analyser-api .
docker run -p 8000:8000 resume-analyser-api
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `422`: Validation error (invalid input)
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

Error response format:
```json
{
  "detail": "Error message here"
}
```

## Limitations

- **Input Length:** Resumes are truncated to 200 tokens
- **Language:** Optimized for English resumes
- **Batch Size:** Maximum 50 resumes per batch request
- **Rate Limiting:** Not implemented (add if needed for production)

## Troubleshooting

### Model not loading
- Ensure you have internet connection for first-time model download
- Check if you have enough disk space (~1GB for model)
- Verify transformers library is installed correctly

### CUDA out of memory
- Reduce batch size in batch predictions
- Use CPU mode if GPU memory is insufficient
- Close other GPU-intensive applications

### Slow inference
- Use GPU for faster inference
- Enable batch processing for multiple resumes
- Consider model quantization for production

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- GitHub: [Swakyxd/Resume-Analyser](https://github.com/Swakyxd/Resume-Analyser)
- Model: [HuggingFace Hub](https://huggingface.co/SwaKyxd/resume-analyser-bert)
